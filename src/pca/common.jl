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
    s = ArgParseSettings("Principal component analysis MPI kernel", autofix_names=true)
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
            range_tester = (x) -> x >= 1            
        "--ncomponents"
            help = "Number of principal components to compute (defaults to computing all principal components)"           
            arg_type = Int            
            range_tester = (x) -> x >= 1
        "--nreplicas"
            help = "Number of replicas of each data partition"
            default = 1
            arg_type = Int
            range_tester = (x) -> x >= 1            
        "--nwait"
            help = "Number of replicas to wait for in each iteration (defaults to all replicas)"
            arg_type = Int
            range_tester = (x) -> x >= 1
        "--inputdataset"
            help = "Input dataset name"
            default = "X"
            arg_type = String            
        "--outputdataset"
            help = "Output dataset name"
            default = "V"
            arg_type = String
        "--saveiterates"
            help = "Save all intermediate iterates to the output file"
            action = :store_true
    end

    # optionally add implementation-specific arguments
    if @isdefined update_argsettings!
        update_argsettings!(s)
    end

    # common parsing
    parsed_args = parse_args(s, as_symbols=true)

    # optional implementation-specific parsing
    if @isdefined update_parsed_args!
        update_parsed_args!(s, parsed_args)
    end

    return parsed_args    
end

"""

Main loop run by each worker.
"""
function worker_loop(localdata, recvbuf, sendbuf; kwargs...)
    
    # control channel, to tell the workers when to exit
    crreq = MPI.Irecv!(zeros(1), root, control_tag, comm)

    # first iteration (initializes state)
    rreq = MPI.Irecv!(recvbuf, root, data_tag, comm)
    index, _ = MPI.Waitany!([crreq, rreq])
    if index == 1 # exit message on control channel
        return
    end            
    state = worker_task!(recvbuf, sendbuf, localdata; kwargs...)
    MPI.Isend(sendbuf, root, data_tag, comm)

    # remaining iterations
    while true
        rreq = MPI.Irecv!(recvbuf, root, data_tag, comm)
        index, _ = MPI.Waitany!([crreq, rreq])
        if index == 1 # exit message on control channel
            break
        end
        state = worker_task!(recvbuf, sendbuf, localdata; state=state, kwargs...)
        MPI.Isend(sendbuf, root, data_tag, comm)
    end
    return
end

function worker_main()
    nworkers = MPI.Comm_size(comm) - 1
    parsed_args = parse_commandline(isroot)
    nsamples, dimension = problem_size(parsed_args[:inputfile], parsed_args[:inputdataset])
    try
        localdata, recvbuf, sendbuf = worker_setup(rank, nworkers; parsed_args...)
        worker_loop(localdata, recvbuf, sendbuf; kwargs=parsed_args)
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

function coordinator_main()

    # setup
    parsed_args = parse_commandline(isroot)
    nsamples, dimension = problem_size(parsed_args[:inputfile], parsed_args[:inputdataset])
    nworkers = MPI.Comm_size(comm) - 1
    0 < nworkers <= nsamples || throw(DomainError(nworkers, "The number of workers must be in [1, nsamples]"))
    parsed_args[:nworkers] = nworkers
    ncomponents::Int = isnothing(parsed_args[:ncomponents]) ? dimension : parsed_args[:ncomponents]
    parsed_args[:ncomponents] = ncomponents
    ncomponents <= dimension || throw(DimensionMismatch("ncomponents is $ncomponents, but the dimension is $dimension"))
    niterations::Int = parsed_args[:niterations]
    niterations > 0 || throw(DomainError(niterations, "The number of iterations must be non-negative"))
    saveiterates::Bool = parsed_args[:saveiterates]
    nreplicas::Int = parsed_args[:nreplicas]
    mod(nworkers, nreplicas) == 0 || throw(ArgumentError("nworkers must be divisible by nreplicas"))
    npartitions = div(nworkers, nreplicas)
    nwait::Int = isnothing(parsed_args[:nwait]) ? npartitions : parsed_args[:nwait]
    parsed_args[:nwait] = nwait
    0 < nwait <= npartitions || throw(DomainError(nwait, "nwait must be in [1, npartitions]"))

    # worker pool and communication buffers
    pool = StragglerPool(nworkers)
    V, recvbuf, sendbuf = coordinator_setup(nworkers; parsed_args...)
    mod(length(recvbuf), nworkers) == 0 || error("the length of recvbuf must be divisible by the number of workers")
    ∇ = similar(V)
    ∇ .= 0
    isendbuf = similar(sendbuf, nworkers*length(sendbuf))
    irecvbuf = similar(recvbuf)    

    # views into recvbuf corresponding to each worker
    n = div(length(recvbuf), nworkers)
    recvbufs = [view(recvbuf, (i-1)*n+1:i*n) for i in 1:nworkers]

    # optionally store all intermediate iterates
    if saveiterates
        iterates = zeros(size(V)..., niterations)
    else
        iterates = zeros(size(V)..., 0)
    end

    # store which workers responded in each iteration
    responded = zeros(Bool, nworkers, niterations)

    # to record iteration time
    ts_compute = zeros(niterations)
    ts_update = zeros(niterations)

    function fwait(epoch, repochs)
        length(repochs) == nworkers || throw(DomainError(nworkers, "repochs must have length nworkers"))
        rreplicas = 0
        for partition in 1:npartitions
            for replica in 1:nreplicas
                i = (partition-1)*nreplicas + replica
                if repochs[i] == epoch
                    rreplicas += 1
                    break
                end
            end
        end
        rreplicas >= nwait
    end

    # first iteration (initializes state)
    epoch = 1
    ts_compute[epoch] = @elapsed begin
        repochs = kmap!(sendbuf, recvbuf, isendbuf, irecvbuf, fwait, epoch, pool, comm; tag=data_tag)
    end
    responded[:, epoch] .= repochs .== epoch
    ts_update[epoch] = @elapsed begin
        gradient_state = update_gradient!(∇, recvbufs, sendbuf, epoch, repochs; parsed_args...)
        iterate_state = update_iterate!(V, ∇, sendbuf, epoch, repochs; parsed_args...)
    end
    if saveiterates
        iterates[:, :, epoch] .= V
    end

    # remaining iterations
    for epoch in 2:niterations
        ts_compute[epoch] = @elapsed begin
            repochs = kmap!(sendbuf, recvbuf, isendbuf, irecvbuf, fwait, epoch, pool, comm; tag=data_tag)
        end
        responded[:, epoch] .= repochs .== epoch
        ts_update[epoch] = @elapsed begin
            gradient_state = update_gradient!(∇, recvbufs, sendbuf, epoch, repochs; state=gradient_state, parsed_args...)
            iterate_state = update_iterate!(V, ∇, sendbuf, epoch, repochs; state=iterate_state, parsed_args...)            
        end
        if saveiterates
            iterates[:, :, epoch] .= V
        end
    end

    shutdown(pool)
    h5open(parsed_args[:outputfile], "w") do fid

        # write parameters to the output file
        for (key, val) in parsed_args
            fid["parameters/$key"] = val
        end

        # write the computed principal components
        fid[parsed_args[:outputdataset]] = V

        # optionally save all iterates
        if saveiterates
            fid["iterates"] = iterates
        end

        # write benchmark data
        fid["benchmark/t_compute"] = ts_compute
        fid["benchmark/t_update"] = ts_update
        fid["benchmark/responded"] = responded
    end
    return
end

if isroot
    coordinator_main()
else
    worker_main()
end
MPI.Barrier(comm)