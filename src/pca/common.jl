using MPI, MPIAsyncPools
using CodedComputing
using HDF5, LinearAlgebra
using MKLSparse
using ArgParse

MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0
const rank = MPI.Comm_rank(comm)
const isroot = MPI.Comm_rank(comm) == root
const data_tag = 0
const control_tag = 1
const COMMON_BYTES = 8 # number of bytes reserved for communication from worker_main to coordinator_main

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
        "--enablegc"
            help = "Enable garbage collection while running the computation (defaults to disabled)"
            action = :store_true
    end

    # optionally add implementation-specific arguments
    if @isdefined update_argsettings!
        update_argsettings!(s)
    end

    # common parsing
    parsed_args = parse_args(s, as_symbols=true)

    # store the number of workers
    nworkers = MPI.Comm_size(comm) - 1
    parsed_args[:nworkers] = nworkers    

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
    t = @elapsed state = worker_task!(recvbuf, sendbuf, localdata; kwargs...)
    reinterpret(Float64, view(sendbuf, 1:COMMON_BYTES))[1] = t # send the recorded compute latency to the coordinator    
    MPI.Isend(sendbuf, root, data_tag, comm)

    # remaining iterations
    while true
        rreq = MPI.Irecv!(recvbuf, root, data_tag, comm)
        index, _ = MPI.Waitany!([crreq, rreq])
        if index == 1 # exit message on control channel
            break
        end
        t = @elapsed state = worker_task!(recvbuf, sendbuf, localdata; state=state, kwargs...)
        reinterpret(Float64, view(sendbuf, 1:COMMON_BYTES))[1] = t # send the recorded compute latency to the coordinator
        MPI.Isend(sendbuf, root, data_tag, comm)
    end
    return
end

"""

Main function run by each worker.
"""
function worker_main()
    nworkers = MPI.Comm_size(comm) - 1
    parsed_args = parse_commandline(isroot)
    try
        localdata, recvbuf, sendbuf = worker_setup(rank, nworkers; parsed_args...)
        GC.gc()
        GC.enable(parsed_args[:enablegc])
        MPI.Barrier(comm)        
        worker_loop(localdata, recvbuf, sendbuf; parsed_args...)
    catch e
        print(Base.stderr, "rank $rank exiting due to $e")
        exit(0) # only the root exits with non-zero status in case of error
    end
end

"""

Send a stop signal to each worker.
"""
function shutdown(pool::MPIAsyncPool)
    for i in pool.ranks
        MPI.Isend(zeros(1), i, control_tag, comm)
    end
end

"""

Main loop run by the coordinator.
"""
function coordinator_main()

    # setup
    parsed_args = parse_commandline(isroot)
    niterations::Int = parsed_args[:niterations]
    saveiterates::Bool = parsed_args[:saveiterates]
    nworkers::Int = parsed_args[:nworkers]
    @info "Coordinator started"

    # create the output directory if it doesn't exist, and make sure we can write to the output file
    mkpath(dirname(parsed_args[:outputfile]))
    close(h5open(parsed_args[:outputfile], "w"))

    # worker pool and communication buffers
    pool = MPIAsyncPool(nworkers)
    V, recvbuf, sendbuf = coordinator_setup(nworkers; parsed_args...)
    mod(length(recvbuf), nworkers) == 0 || error("the length of recvbuf must be divisible by the number of workers")
    ∇ = similar(V)
    ∇ .= 0
    isendbuf = similar(sendbuf, nworkers*length(sendbuf))
    irecvbuf = similar(recvbuf)

    # views into recvbuf corresponding to each worker
    recvbufs = [view(recvbuf, partition(length(recvbuf), nworkers, i)) for i in 1:nworkers]

    # optionally store all intermediate iterates
    if saveiterates
        iterates = zeros(size(V)..., niterations)
    else
        iterates = zeros(size(V)..., 0)
    end

    # benchmark/analysis data
    responded = zeros(Int, nworkers, niterations)       # which workers respond in each iteration
    latency = zeros(nworkers, niterations)              # per-iteration latency of each worker
    compute_latency = zeros(nworkers, niterations)      # per-iteration computation latency of each worker
    ts_compute = zeros(niterations)                     # overall per-iteration latency
    ts_update = zeros(niterations)                      # per-iteration latency of the coordinator update

    # 2-argument fwait needed for asyncmap!
    f = (epoch, repochs) -> fwait(epoch, repochs; parsed_args...)

    # manually call the GC now, and optionally turn off GC, to avoid pauses later during execution
    GC.gc()
    GC.enable(parsed_args[:enablegc])

    # ensure all workers have finished compiling before starting the computation
    # (this is only necessary when benchmarking)
    MPI.Barrier(comm)
    @info "Optimization starting"

    # first iteration (initializes state)
    epoch = 1
    ts_compute[epoch] = @elapsed begin
        repochs = asyncmap!(pool, sendbuf, recvbuf, isendbuf, irecvbuf, comm, nwait=f, epoch=epoch, tag=data_tag)
    end
    responded[:, epoch] .= repochs
    latency[:, epoch] .= pool.latency
    for i in 1:nworkers
        t = repochs[i] == epoch ? reinterpret(Float64, view(recvbufs[i], 1:COMMON_BYTES))[1] : NaN
        compute_latency[i, epoch] = t
    end
    ts_update[epoch] = @elapsed begin
        state = coordinator_task!(V, ∇, recvbufs, sendbuf, epoch, repochs; parsed_args...)    
    end
    if saveiterates
        ndims = length(size(V))
        selectdim(iterates, ndims+1, epoch) .= V
    end

    # remaining iterations
    for epoch in 2:niterations
        ts_compute[epoch] = @elapsed begin
            repochs = asyncmap!(pool, sendbuf, recvbuf, isendbuf, irecvbuf, comm, nwait=f, epoch=epoch, tag=data_tag)
        end
        responded[:, epoch] .= repochs
        latency[:, epoch] .= pool.latency
        for i in 1:nworkers
            t = repochs[i] == epoch ? reinterpret(Float64, view(recvbufs[i], 1:COMMON_BYTES))[1] : NaN
            compute_latency[i, epoch] = t
        end
        ts_update[epoch] = @elapsed begin
            state = coordinator_task!(V, ∇, recvbufs, sendbuf, epoch, repochs; state, parsed_args...)    
        end
        if saveiterates
            ndims = length(size(V))
            selectdim(iterates, ndims+1, epoch) .= V
        end
    end

    @info "Optimization finished; writing output to disk"

    # signal all workers to stop
    shutdown(pool)

    # write output
    h5open(parsed_args[:outputfile], "w") do fid

        # write program parameters
        for (key, val) in parsed_args
            if isnothing(val)
                continue
            end
            fid["parameters/$key"] = val
        end

        # write the computation output
        fid[parsed_args[:outputdataset]] = V

        # optionally save all iterates
        if saveiterates
            fid["iterates"] = iterates
        end

        # write benchmark data
        fid["benchmark/t_compute"] = ts_compute
        fid["benchmark/t_update"] = ts_update
        fid["benchmark/responded"] = responded
        fid["benchmark/latency"] = latency
        fid["benchmark/compute_latency"] = compute_latency
    end
    @info "Output written to disk; exiting"
    return
end

if isroot

    # check that all necessary functions are defined
    (@isdefined coordinator_setup) || error("function coordinator_setup must be defined")
    (@isdefined coordinator_task!) || error("function coordinator_task! must be defined")    
    (@isdefined worker_setup) || error("function worker_setup must be defined")    
    (@isdefined worker_task!) || error("function worker_task! must be defined")
    (@isdefined fwait) || error("function fwait must be defined")

    # start the coordinator loop
    coordinator_main()
else
    worker_main()
end
MPI.Barrier(comm)