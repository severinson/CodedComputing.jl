using CodedComputing, MPI, MPIStragglers
using HDF5, LinearAlgebra
using ArgParse

# TODO:
# check for type instability
# implementation using Bcast/Gather

MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0
const rank = MPI.Comm_rank(comm)
const isroot = MPI.Comm_rank(comm) == root
const data_tag = 0
const control_tag = 1

"""

Parse command line arguments. One can add implementation-specific arguments or parsing by
defining the functions `update_argsettings!` and `update_parsed_args!`, respectively. See
the function body for details. Returns a dictionary containing the parsed arguments.
"""
function parse_commandline(isroot::Bool)

    # setup error handling so that only the root node prints error messages
    s = ArgParseSettings("Principal component analysis MPI kernel")
    function root_handler(settings::ArgParseSettings, err, err_code::Integer=1)
        println(Base.stderr, err.text)
        println(Base.stderr, usage_string(settings))
        MPI.Finalize()
        exit(err_code)
    end
    function worker_handler(settings::ArgParseSettings, err)
        MPI.Finalize()
        exit(0)
    end    
    if isroot
        s.exc_handler = root_handler
    else
        s.exc_handler = worker_handler
        s.add_help = false
    end

    # command-line arguments common to all pca implementations
    @add_arg_table s begin
        "inputfile"
            help = "HDF5 file containing the input data set"
            required = true
            arg_type = String
            range_tester = ishdf5
        "outputfile"
            help = "HFD5 file to write the output to (defaults to same as inputfile)"
            arg_type = String
            range_tester = (x) -> isnothing(x) || !isfile(x) || ishdf5(x)
        "--niterations"
            help = "Number of iterations to run the algorithm for"
            default = 10
            arg_type = Int            
        "--ncomponents"
            help = "Number of principal components to compute (defaults to computing all principal components)"           
            arg_type = Int            
        "--inputdataset"
            help = "Input dataset name"
            default = "X"
            arg_type = String            
        "--outputdataset"
            help = "Output dataset name"
            default = "V"
            arg_type = String
        "--benchmarkfile"
            help = "HDF5 file to write benchmark data to"
            arg_type = String
    end

    # optionally add implementation-specific arguments
    if @isdefined update_argsettings!
        update_argsettings!(s)
    end

    # common parsing
    parsed_args = parse_args(s)
    if isnothing(parsed_args["outputfile"])
        parsed_args["outputfile"] = parsed_args["inputfile"]        
    end

    # optional implementation-specific parsing
    if @isdefined update_parsed_args!
        update_parsed_args!(s, parsed_args)
    end

    return parsed_args    
end

"""

Main loop run by each worker.
"""
function worker_loop(localdata, dimension::Integer, ncomponents::Integer)
    
    # control channel, to tell the workers when to exit
    crreq = MPI.Irecv!(zeros(1), root, control_tag, comm)

    # working memory
    Vrecv = Matrix{Float64}(undef, dimension, ncomponents)
    Vsend = Matrix{Float64}(undef, dimension, ncomponents)

    # first iteration (initializes state)
    rreq = MPI.Irecv!(Vrecv, root, data_tag, comm)
    index, _ = MPI.Waitany!([crreq, rreq])
    if index == 1 # exit message on control channel
        return
    end            
    state = worker_task!(Vrecv, localdata)
    Vsend .= Vrecv
    MPI.Isend(Vsend, root, data_tag, comm)

    # remaining iterations
    while true
        rreq = MPI.Irecv!(Vrecv, root, data_tag, comm)
        index, _ = MPI.Waitany!([crreq, rreq])
        if index == 1 # exit message on control channel
            break
        end        
        worker_task!(Vrecv, localdata, state)
        Vsend .= Vrecv
        MPI.Isend(Vsend, root, data_tag, comm)
    end
    return
end

function worker_main()
    nworkers = MPI.Comm_size(comm) - 1
    parsed_args = parse_commandline(isroot)
    nsamples, dimension = problem_size(parsed_args["inputfile"], parsed_args["inputdataset"])
    try        
        # read input data for this worker
        localdata = read_localdata(parsed_args["inputfile"], parsed_args["inputdataset"], rank, nworkers)

        # default to computing all principal components
        if isnothing(parsed_args["ncomponents"])
            ncomponents = dimension
        else
            ncomponents::Int = parsed_args["ncomponents"]
        end

        # run the algorithm
        worker_loop(localdata, dimension, ncomponents)
    catch e
        print(Base.stderr, "rank $rank exiting due to $e")
        exit(0) # only the root exits with non-zero status in case of error
    end
end

function shutdown(pool::StragglerPool)
    for i in pool.ranks
        MPI.Isend(zeros(1), i, control_tag, comm)
    end
end

function root_main()

    # setup
    parsed_args = parse_commandline(isroot)
    nsamples, dimension = problem_size(parsed_args["inputfile"], parsed_args["inputdataset"])
    if isnothing(parsed_args["ncomponents"]) # default to computing all principal components
        ncomponents = dimension
    else
        ncomponents::Int = parsed_args["ncomponents"]
    end
    ncomponents <= dimension || throw(DimensionMismatch("ncomponents is $ncomponents, but the dimension is $dimension"))
    niterations::Int = parsed_args["niterations"]
    niterations > 0 || throw(DomainError(niterations, "The number of iterations must be non-negative"))
    nworkers = MPI.Comm_size(comm) - 1
    0 < nworkers <= nsamples || throw(DomainError(nworkers, "The number of workers must be in [1, nsamples]"))

    # worker pool and communication buffers
    pool = StragglerPool(nworkers)
    sendbuf = Matrix{Float64}(undef, dimension, ncomponents)    
    recvbuf = Matrix{Float64}(undef, dimension, nworkers*ncomponents)
    isendbuf = similar(sendbuf, nworkers*length(sendbuf))
    irecvbuf = similar(recvbuf)

    # iterate, initialized at random
    V = view(sendbuf, :, :)
    V .= randn(dimension, ncomponents)
    ∇ = similar(V)

    # results computed by the workers
    Vs = [view(recvbuf, :, (i-1)*ncomponents+1:i*ncomponents) for i in 1:nworkers]

    # main loop
    ts_compute = zeros(niterations)
    ts_update = zeros(niterations)
    for epoch in 1:niterations
        ts_compute[epoch] = @elapsed begin
            epochs = kmap!(sendbuf, recvbuf, isendbuf, irecvbuf, nworkers, epoch, pool, comm; tag=data_tag)        
        end
        ts_update[epoch] = @elapsed begin
            update_gradient!(∇, Vs, epochs .== epoch)
            update_iterate!(V, ∇)
        end
    end

    shutdown(pool)

    # write the computed principal components to disk
    # outputfile::String = parsed_args["outputfile"] 
    # outputdataset::String = parsed_args["outputdataset"]
    try_write(sendbuf, parsed_args["outputfile"], parsed_args["outputdataset"])
    # if isfile(outputfile)
    #     mode = "r+"
    # else
    #     mode = "w"
    # end
    # jldopen(outputfile, mode, compress=true) do file
    #     if outputdataset in names(file)
    #         delete!(file, outputdataset)
    #     end
    #     file[outputdataset] = sendbuf # aliased to V, writing a view results in a crash
    # end   
    
    # write benchmark data to disk (if a benchmark file was provided)
    if !isnothing(parsed_args["benchmarkfile"])
        try_write(ts_compute, parsed_args["benchmarkfile"], "ts_compute")    
        try_write(ts_update, parsed_args["benchmarkfile"], "ts_update")    
        # benchmarkfile::String = parsed_args["benchmarkfile"]
        # if isfile(benchmarkfile)
        #     mode = "r+"
        # else
        #     mode = "w"
        # end        
        # jldopen(benchmarkfile, mode, compress=true) do file    
        #     if "ts_compute" in names(file)
        #         delete!(file, "ts_compute")
        #     end
        #     file["ts_compute"] = ts_compute
        #     if "ts_update" in names(file)
        #         delete!(file, "ts_update")
        #     end
        #     file["ts_update"] = ts_update    
        # end
    end
    
    return
end

function try_write(data::Array, filename::String, dataset::String; replace=true, rethrow_exception=false)    
    try
        h5open(filename, "cw") do file
            if dataset in names(file)
                if replace
                    delete!(file, dataset)
                    file[dataset] = data
                end
            else
                file[dataset] = data
            end
        end
    catch e
        print(Base.stderr, "Writing to $(filename):$dataset failed with exception $e\n")
        stacktrace(catch_backtrace())
        if rethrow_exception
            rethrow()
        end
    end
end

if isroot
    root_main()
    # MPI.Barrier(comm)
    # root_main()
else
    worker_main()
    # MPI.Barrier(comm)
    # worker_main()
end
MPI.Barrier(comm)