using Random
using MPI, MPIAsyncPools
using CodedComputing
using HDF5, LinearAlgebra
using ArgParse
using Dates
using Distributions
using DataStructures

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
        "--nsavediterates"
            help = "Maximum number of iterates to save to disk"
            default = 20
            arg_type = Int            
            range_tester = (x) -> 0 < x
        "--enablegc"
            help = "Enable garbage collection while running the computation (defaults to disabled)"
            action = :store_true
        "--loadbalance"
            help = "Enable the load-balancer"
            action = :store_true
        "--lbtimelimit"
            help = "Time limit in seconds of the load-balancing optimization"
            default = 1.0
            arg_type = Float64
            range_tester = (x) -> 0 < x
        "--lbminimprovement"
            help = "Minimum improvement required for the load-balancer to accept a new solution"
            default = 0.98
            arg_type = Float64
            range_tester = (x) -> x <= 1
        "--lbaggressive"
            help = "[DEPRECATED] Set the load-balancer in aggressive mode, in which it doesn't make workers slower"
            action = :store_true
        "--profilerwindowsize"
            help = "Number of seconds that latency statistics are computed over"
            default = 10
            arg_type = Int
            range_tester = (x) -> 0 < x
        "--profilerminsamples"
            help = "Minimum number of samples required for the profiler to compute mean and variance"
            default = 10
            arg_type = Int
            range_tester = (x) -> 1 < x
        "--randomseed"
            help = "Used to compute a deterministic random seed that is unique to the coordinator and each worker"
            arg_type = UInt
            default = rand(UInt)
        "--nslow"
            help = "Number of workers that are slowed down artificially in each iteration"
            arg_type = Int
            default = 0          
            range_tester = (x) -> 0 <= x
        "--slowprob"
            help = "Each worker is artificially slowed down with this probability (works in addition to --nslow)"
            arg_type = Float64
            default = 0.0
            range_tester = (x) -> 0 <= x <= 1
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

Spin on `time_ns()` for the specified number of seconds before returning.
"""
function highres_sleep(seconds)
    t0 = time_ns()
    elapsed = 0.0
    while elapsed < seconds
        elapsed = (time_ns() - t0) / 1e9
    end
    return
end

"""

Main loop run by each worker.
"""
function worker_loop(localdata, recvbuf, sendbuf; nslow::Integer, slowprob::Real, nworkers::Integer, kwargs...)

    @info "Worker $rank is running on $(gethostname())"

    # Markov model artificial latency
    ## timestamp of the most recent Markov model state transition
    transition_time = time_ns()

    ## model
    burst_latency_increase = 1.103
    transition_interval = 10 # seconds
    P = [0.989 0.011; 0.286 0.714]

    ## current state
    ## (initialized according to the steady state distribution)
    burst_state = 1
    if rand() < 0.037037
        burst_state = 2
    end    
    
    # control channel, to tell the workers when to exit
    crreq = MPI.Irecv!(zeros(1), root, control_tag, comm)

    # first iteration (initializes state)
    rreq = MPI.Irecv!(recvbuf, root, data_tag, comm)
    index, _ = MPI.Waitany!([crreq, rreq])
    if index == 1 # exit message on control channel
        return
    end
    t = @elapsed state = worker_task!(recvbuf, sendbuf, localdata; nworkers, kwargs...)
    reinterpret(Float64, view(sendbuf, 1:COMMON_BYTES))[1] = t # send the recorded compute latency to the coordinator    
    MPI.Isend(sendbuf, root, data_tag, comm)

    # remaining iterations
    i = 1
    start_time = time_ns()
    while true

        # # make a state transition every transition_interval seconds
        # current_time = time_ns()
        # if (current_time - transition_time) / 1e9 >= transition_interval
        #     transition_time = current_time
        #     if rand() < P[burst_state, 1]
        #         if burst_state == 2
        #             @info "worker $rank transitioning 2 => 1"
        #         end
        #         burst_state = 1
        #     else
        #         if burst_state == 1
        #             @info "worker $rank transitioning 1 => 2"
        #         end                
        #         burst_state = 2
        #     end
        # end

        # worker task
        rreq = MPI.Irecv!(recvbuf, root, data_tag, comm)
        index, _ = MPI.Waitany!([crreq, rreq])
        if index == 1 # exit message on control channel
            MPI.Cancel!(rreq) # mark the rreq for cancellation
            MPI.Test!(rreq) # cleanup the data receive request            
            break
        end
        t = @elapsed state = worker_task!(recvbuf, sendbuf, localdata; state, nworkers, kwargs...)
        # t = @elapsed begin
        #     t0 = @elapsed state = worker_task!(recvbuf, sendbuf, localdata; state, nworkers, kwargs...)
            
        #     # simulate using a slower computer
        #     latency_increase_factor = 1
        #     if latency_increase_factor > 1
        #         highres_sleep(t0 * (latency_increase_factor - 1))
        #     end

        #     # introduce artificial latency variation between workers
        #     seconds = 0.0
        #     if (time_ns() - start_time) / 1e9 < 1 || rank < 40
        #         seconds = t0 * latency_increase_factor * (0.4 * rank / nworkers)
        #     end
        #     highres_sleep(seconds)

        #     # # during the first 25 iterations (corresponding to one pass over the data), slow down 
        #     # # the first 9 workers by 25% more to simulate a latency burst
        #     # if i < 25 && rank < 10
        #     #     highres_sleep((t0 + t0 * 0.4 * rank / nworkers) * 0.25)
        #     # end

        #     # # if in the latency burst state, sleep to make up the latency difference
        #     # if burst_state == 2
        #     #     highres_sleep(t0 * (burst_latency_increase - 1))
        #     # end

        #     # # the nslow first workers are artificially slowed down
        #     # # (counted as comp. delay)
        #     # if rank <= nslow
        #     #     highres_sleep(t0)
        #     # end

        #     # # workers 1-3 are slow for the first 100 iterations
        #     # # the remaining workers are always slow
        #     # if i < 100 || rank > 3
        #     #     highres_sleep(t0)
        #     # end

        #     # # workers 4-6 become even slower after 50 iterations
        #     # if i >= 50 && 3 < rank <= 6
        #     #     highres_sleep(t0)
        #     # end
        # end

        # # workers are artificially slowed down with prob. slowprob.
        # # (counted as comm. delay)
        # if !iszero(slowprob) && rand() < slowprob
        #     highres_sleep(t0)
        # end

        # send response to coordinator
        reinterpret(Float64, view(sendbuf, 1:COMMON_BYTES))[1] = t # send the recorded compute latency to the coordinator
        MPI.Send(sendbuf, root, data_tag, comm)
        i += 1
    end
    return
end

"""

Main function run by each worker.
"""
function worker_main()
    nworkers = MPI.Comm_size(comm) - 1
    parsed_args = parse_commandline(isroot)
    Random.seed!(parsed_args[:randomseed] + rank)    
    try
        localdata, recvbuf, sendbuf = worker_setup(rank, nworkers; parsed_args...)
        GC.gc()
        GC.enable(parsed_args[:enablegc])
        @info "(rank $rank) ready"
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
        MPI.Send(zeros(1), i, control_tag, comm)
    end
end

"""

Select the next partition to process for the `i`-th worker.
"""
function select_partitions!(partition_indices::AbstractVector{<:Integer}, nsubpartitions_all::AbstractVector{<:Integer}, i::Integer)
    0 < i <= length(partition_indices) || throw(DimensionMismatch("i is $i, but partition_indices has dimension $(length(partition_indices))"))
    0 < i <= length(nsubpartitions_all) || throw(DimensionMismatch("i is $i, but nsubpartitions_all has dimension $(length(nsubpartitions_all))"))
    0 < nsubpartitions_all[i] || throw(ArgumentError("Invalid number of partitions for worker $i: $nsubpartitions_all"))
    partition_indices[i] = mod(partition_indices[i], nsubpartitions_all[i]) + 1
end

"""

For each worker, choose the sub-partition it should process.
"""
function select_partitions!(partition_indices::AbstractVector{<:Integer}, nsubpartitions_all::AbstractVector{<:Integer})
    length(partition_indices) == length(nsubpartitions_all) || throw(DimensionMismatch("partition_indices has dimension $(length(partition_indices)), but nsubpartitions_all has dimension $(size(nsubpartitions_all))"))    
    for i in 1:length(partition_indices)
        select_partitions!(partition_indices, nsubpartitions_all, i)
    end
    partition_indices
end

"""

For each worker, write the number of sub-partitions and the index of the sub-partition to process to `sendbuf`.
"""
function write_partitions!(sendbuf::AbstractVector{UInt8}; partition_indices::AbstractVector{<:Integer}, nsubpartitions_all::AbstractVector{<:Integer})
    length(partition_indices) == length(nsubpartitions_all) || throw(DimensionMismatch("partition_indices has dimension $(length(partition_indices)), but nsubpartitions_all has dimension $(size(nsubpartitions_all))"))        
    nworkers = length(partition_indices)
    s = sizeof(UInt16) * 2 * nworkers
    s <= length(sendbuf) || throw(DimensionMismatch("sendbuf is of length $(length(sendbuf)), but nworkers is $nworkers"))
    vs = reinterpret(Tuple{UInt16,UInt16}, view(sendbuf, 1:s))
    for i in 1:nworkers
        vs[i] = (nsubpartitions_all[i], 0) # sub-partition_index is chosen by the workers
    end
    sendbuf
end

"""

Return the metadata recorded in a buffer received from a worker.
"""
function metadata_from_recvbuf(recvbuf::AbstractVector{UInt8})
    metadata = reinterpret(UInt16, metadata_view(recvbuf))
    length(metadata) == 4 || throw(ArgumentError("metadata is $metadata"))
    canary, worker_rank, nsubpartitions, subpartition_index = metadata
    canary == CANARY_VALUE || throw(ArgumentError("metadata is $metadata"))
    worker_rank, nsubpartitions, subpartition_index
end

"""

Return the compute delay recorded in a buffer received from a worker.
"""
function comp_delay_from_recvbuf(recvbuf::AbstractVector{UInt8})
    reinterpret(Float64, view(recvbuf, 1:COMMON_BYTES))[1]
end

"""

Main loop run by the coordinator.
"""
function coordinator_main()

    # setup
    latency_tol = 0.02
    parsed_args = parse_commandline(isroot)
    niterations::Int = parsed_args[:niterations]
    saveiterates::Bool = parsed_args[:saveiterates]
    nworkers::Int = parsed_args[:nworkers]
    nwait::Int = parsed_args[:nwait]
    loadbalance::Bool = parsed_args[:loadbalance]
    nsubpartitions_all = fill(parsed_args[:nsubpartitions], nworkers)
    partition_indices = zeros(UInt16, nworkers)
    to_worker_common_bytes = sizeof(UInt16) * 2 * nworkers
    prev_repochs = zeros(Int, nworkers)
    Random.seed!(parsed_args[:randomseed])
    @info "Coordinator is running with $(Threads.nthreads()) threads on $(gethostname())"
    @info "Main task is running on thread $(Threads.threadid())"

    # create the output directory if it doesn't exist, and make sure we can write to the output file
    mkpath(dirname(parsed_args[:outputfile]))
    close(h5open(parsed_args[:outputfile], "w"))

    # worker pool and communication buffers
    pool = MPIAsyncPool(nworkers)
    V, recvbuf, sendbuf, state, gradient_recvf = coordinator_setup(nworkers; parsed_args...)
    mod(length(recvbuf), nworkers) == 0 || error("the length of recvbuf must be divisible by the number of workers")
    ∇ = similar(V)
    ∇ .= 0
    isendbuf = similar(sendbuf, nworkers*length(sendbuf))
    irecvbuf = similar(recvbuf)

    # views into recvbuf corresponding to each worker
    recvbufs = [view(recvbuf, partition(length(recvbuf), nworkers, i)) for i in 1:nworkers]

    # number of iterates to save
    nsavediterates::Int = parsed_args[:nsavediterates]
    savediterateindices = unique(round.(Int, exp.(range(log(1), log(niterations), length=nsavediterates))))
    nsavediterates = length(savediterateindices)
    savediterateindex = 1

    # optionally store intermediate iterates
    if saveiterates
        iterates = zeros(Float64, size(V)..., nsavediterates)
    else
        iterates = zeros(Float64, size(V)..., 0)
    end

    # benchmark/analysis data
    responded = zeros(Int, nworkers, niterations)       # which workers respond in each iteration
    latency = zeros(nworkers, niterations)              # per-iteration latency of each worker
    compute_latency = zeros(nworkers, niterations)      # per-iteration computation latency of each worker
    ts_compute = zeros(niterations)                     # overall per-iteration latency
    ts_update = zeros(niterations)                      # per-iteration latency of the coordinator update
    loadbalanced = zeros(niterations)                   # indicates which iterations the load-balancer made changes in

    # 2-argument fwait needed for asyncmap!
    f = (epoch, repochs) -> fwait(epoch, repochs; parsed_args...)

    # 4-argument recvf needed for asyncmap!
    recvf = isnothing(gradient_recvf) ? nothing : (args...) -> gradient_recvf(args...; state, parsed_args...)

    # manually call the GC now, and optionally turn off GC, to avoid pauses later during execution
    GC.gc()
    GC.enable(parsed_args[:enablegc])

    # setup the latency profiler
    profiler_chin, profiler_chout = CodedComputing.setup_profiler_channels()
    profiler_task = Threads.@spawn CodedComputing.latency_profiler(
        profiler_chin, profiler_chout; 
        nworkers, 
        windowsize=Second(parsed_args[:profilerwindowsize]),
        minsamples=parsed_args[:profilerminsamples],
        )

    # vector of communication latencies for each worker
    # (to use instead of the measured value for stale results)
    communication_latency = fill(NaN, nworkers)
    communication_latency_samples = CircularBuffer{Float64}(nworkers)

    # setup the load-balancer
    _, loadbalancer_chout = CodedComputing.setup_loadbalancer_channels()
    loadbalancer_task = Threads.@spawn CodedComputing.load_balancer(
        profiler_chout, loadbalancer_chout;
        nsubpartitions=parsed_args[:nsubpartitions],
        nwait, nworkers,
        min_improvement=parsed_args[:lbminimprovement],
        )

    # ensure all workers have finished compiling before starting the computation
    # (this is only necessary when benchmarking)
    @info "Waiting for workers to get ready..."
    MPI.Barrier(comm)
    @info "Optimization starting"

    # remaining iterations
    for epoch in 1:niterations

        # set to NaN to remove old latency results
        pool.latency .= NaN

        # gradient_state, iterate_state = state
        # ∇i, tg = gradient_state
        # @info "epoch $epoch, tg.ninit / tg.n: $(tg.ninit / tg.n)"
        # @info "epoch $epoch"

        ## force an error if the profiler or load-balancer task has failed
        if istaskfailed(profiler_task)
            wait(profiler_task)
        end
        if istaskfailed(loadbalancer_task)
            wait(loadbalancer_task)
        end
        
        ## get updated partitioning from the load-balancer
        while isready(loadbalancer_chout)
            vout = take!(loadbalancer_chout)
            0 < vout.worker <= nworkers || throw(ArgumentError("incorrect $vout"))
            0 < vout.p || throw(ArgumentError("incorrect $vout"))
            if nsubpartitions_all[vout.worker] != vout.p
                # @info "epoch $epoch, worker $(vout.worker) nsubpartitions $(nsubpartitions_all[vout.worker]) => $(vout.p)"
                nsubpartitions_all[vout.worker] = vout.p
            end
            loadbalanced[epoch] = true
        end

        ## update partitioning        
        select_partitions!(partition_indices, nsubpartitions_all)
        write_partitions!(sendbuf; partition_indices, nsubpartitions_all)

        ## worker task
        ts_compute[epoch] = @elapsed begin
            repochs = asyncmap!(pool, sendbuf, recvbuf, isendbuf, irecvbuf, comm; nwait=f, epoch=epoch, tag=data_tag, recvf, latency_tol)
        end
        responded[:, epoch] .= repochs

        ## record latency
        latency[:, epoch] .= pool.latency
        for i in 1:nworkers
            t = repochs[i] == epoch ? reinterpret(Float64, view(recvbufs[i], 1:COMMON_BYTES))[1] : NaN
            compute_latency[i, epoch] = t
        end

        # record new communication latency values
        for i in 1:nworkers
            if repochs[i] == epoch # fresh result
                communication_latency[i] = latency[i, epoch] - compute_latency[i, epoch]
                push!(communication_latency_samples, communication_latency[i])
            end
        end

        ## send any new latency information to the profiling sub-system
        if loadbalance
            timestamp = Time(now())
            for i in 1:nworkers
                if repochs[i] != prev_repochs[i]
                    _, nsubpartitions, subpartition_index = metadata_from_recvbuf(recvbufs[i])            
                    comp_latency = comp_delay_from_recvbuf(recvbufs[i])
                    comm_latency = isnan(communication_latency[i]) ? mean(communication_latency_samples) : communication_latency[i]
                    v = CodedComputing.ProfilerInput(i, 1/nworkers, 1/nsubpartitions, timestamp, comp_latency, comm_latency)
                    push!(profiler_chin, v)
                end
                prev_repochs[i] = repochs[i]
            end
        end      

        ## coordinator task
        ts_update[epoch] = @elapsed begin
            state = coordinator_task!(V, ∇, recvbufs, sendbuf, epoch, repochs; state, parsed_args...)    
        end

        ## optinally record the iterate
        if saveiterates && epoch == savediterateindices[savediterateindex]
            ndims = length(size(V))
            selectdim(iterates, ndims+1, savediterateindex) .= V
            savediterateindex += 1
        end        
    end

    @info "Optimization finished; writing output to disk"

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
            fid["savediterateindices"] = savediterateindices
        end

        # write benchmark data
        fid["benchmark/t_compute"] = ts_compute
        fid["benchmark/t_update"] = ts_update
        fid["benchmark/responded"] = responded
        fid["benchmark/latency"] = latency
        fid["benchmark/compute_latency"] = compute_latency
        fid["benchmark/loadbalanced"] = loadbalanced
    end
    @info "Output written to disk; stopping tasks"

    # signal all workers to stop
    waitall!(pool, recvbuf, irecvbuf)
    shutdown(pool)

    # stop the profiler and load-balancer
    close(profiler_chin)
    close(profiler_chout)
    close(loadbalancer_chout)

    @info "Tasks stopped; exiting"
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