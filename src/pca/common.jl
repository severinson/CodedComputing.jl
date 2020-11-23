using CodedComputing, MPI, MPIStragglers
using HDF5, LinearAlgebra
using ArgParse

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
            range_tester = HDF5.ishdf5
        "outputfile"
            help = "HFD5 file to write the output to"
            required = true
            arg_type = String
            range_tester = (x) -> !isfile(x) || HDF5.ishdf5(x)
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
        "--nwait"
            help = "Number of workers to wait for in each iteration (defaults to all workers)"
            arg_type = Int
        "--saveiterates"
            help = "Save all intermediate iterates to the output file"
            action = :store_true
    end

    # optionally add implementation-specific arguments
    if @isdefined update_argsettings!
        update_argsettings!(s)
    end

    # common parsing
    parsed_args = parse_args(s)

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
    nworkers = MPI.Comm_size(comm) - 1
    0 < nworkers <= nsamples || throw(DomainError(nworkers, "The number of workers must be in [1, nsamples]"))
    parsed_args["nworkers"] = nworkers
    ncomponents::Int = isnothing(parsed_args["ncomponents"]) ? dimension : parsed_args["ncomponents"]
    parsed_args["ncomponents"] = ncomponents
    ncomponents <= dimension || throw(DimensionMismatch("ncomponents is $ncomponents, but the dimension is $dimension"))
    nwait::Int = isnothing(parsed_args["nwait"]) ? nworkers : parsed_args["nwait"]
    parsed_args["nwait"] = nwait
    0 < nwait <= nworkers || throw(DomainError(nwait, "nwait must be in [1, nworkers]"))    
    niterations::Int = parsed_args["niterations"]
    niterations > 0 || throw(DomainError(niterations, "The number of iterations must be non-negative"))
    saveiterates::Bool = parsed_args["saveiterates"]

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

    # optionally store all intermediate iterates
    if saveiterates
        iterates = zeros(dimension, ncomponents, niterations)
    else
        iterates = zeros(dimension, ncomponents, 0)
    end

    # results computed by the workers
    Vs = [view(recvbuf, :, (i-1)*ncomponents+1:i*ncomponents) for i in 1:nworkers]

    # main loop
    ts_compute = zeros(niterations)
    ts_update = zeros(niterations)
    for epoch in 1:niterations
        ts_compute[epoch] = @elapsed begin
            epochs = kmap!(sendbuf, recvbuf, isendbuf, irecvbuf, nwait, epoch, pool, comm; tag=data_tag)
        end
        ts_update[epoch] = @elapsed begin
            update_gradient!(∇, Vs, epochs .== epoch)
            update_iterate!(V, ∇)
        end
        if saveiterates
            iterates[:, :, epoch] .= V
        end
    end

    shutdown(pool)
    h5open(parsed_args["outputfile"], "w") do fid

        # write parameters to the output file
        for (key, val) in parsed_args
            fid["parameters/$key"] = val
        end

        # write the computed principal components
        # sendbuf is aliased to V (writing a view results in a crash)
        fid[parsed_args["outputdataset"]] = sendbuf

        # optionally save all iterates
        if saveiterates
            fid["iterates"] = iterates
        end

        # write benchmark data
        fid["benchmark/ts_compute"] = ts_compute
        fid["benchmark/ts_update"] = ts_update
    end
    return
end

if isroot
    root_main()
else
    worker_main()
end
MPI.Barrier(comm)