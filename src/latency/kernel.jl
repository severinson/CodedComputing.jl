using CodedComputing, MPI, MPIStragglers
using HDF5, LinearAlgebra, SparseArrays
using ArgParse

MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0
const rank = MPI.Comm_rank(comm)
const isroot = MPI.Comm_rank(comm) == root
const data_tag = 0
const control_tag = 1

"""

Parse command line arguments
"""
function parse_commandline(isroot::Bool)

    # setup error handling so that only the root node prints error messages
    s = ArgParseSettings("Latency tracing kernel", autofix_names=true)
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
        "outputfile"
            help = "HFD5 file to write the output to"
            required = true
            arg_type = String
            range_tester = (x) -> !isfile(x) || HDF5.ishdf5(x)        
        "--niterations"
            help = "Number of iterations to run"
            required = true            
            arg_type = Int
            range_tester = (x) -> x >= 1
        "--nbytes"
            help = "Number of bytes communicated in each direction per iteration"
            required = true
            arg_type = Int            
            range_tester = (x) -> x >= 8 # to fit a Float64
        "--nrows"
            help = "Number of rows of the data matrix"
            required = true
            arg_type = Int
            range_tester = (x) -> x >= 1
        "--ncols"
            help = "Number of columns of the data matrix"
            required = true
            arg_type = Int
            range_tester = (x) -> x >= 1            
        "--ncomponents"
            help = "Number of columns of the iterate matrix"
            required = true
            arg_type = Int
            range_tester = (x) -> x >= 1                        
        "--density"
            help = "Density of the data matrix"
            required = true
            arg_type = Float64
            range_tester = (x) -> 0 < x <= 1
        "--nwait"
            help = "Number of workers to wait for in each iteration"
            required = true
            arg_type = Int
            range_tester = (x) -> x >= 1
        "--timeout"
            help = "Amount of time to sleep between iterations"
            required = true
            arg_type = Float64
            range_tester = (x) -> 0 <= x
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

function worker_task!(V, W, X)
    mul!(W, X, V)
    mul!(V, X', W)
end

"""

Receive from the coordinator, perform a matrix-matrix multiplication, and respond with some bytes.
"""
function worker_main()
    parsed_args = parse_commandline(isroot)
    nworkers = MPI.Comm_size(comm) - 1    
    parsed_args[:nworkers] = nworkers    

    # communication setup
    recvbuf = zeros(UInt8, parsed_args[:nbytes])
    sendbuf = zeros(UInt8, parsed_args[:nbytes])
    tbuf = reinterpret(Float64, view(sendbuf, 1:8))

    # worker task setup
    X = sprand(parsed_args[:nrows], parsed_args[:ncols], parsed_args[:density])
    V = randn(parsed_args[:ncols], parsed_args[:ncomponents])
    W = zeros(parsed_args[:nrows], parsed_args[:ncomponents])

    # main loop (niterations+1 to account for the dummy iteration)
    for i in 1:(parsed_args[:niterations]+1)
        rreq = MPI.Irecv!(recvbuf, root, data_tag, comm)
        tbuf[1] = @elapsed worker_task!(V, W, X)
        MPI.Isend(sendbuf, root, data_tag, comm)        
    end
    return
end

"""

Send some bytes to the workers, wait for their response, sleep for a bit, and repeat.
"""
function coordinator_main()    
    parsed_args = parse_commandline(isroot)
    nworkers = MPI.Comm_size(comm) - 1        
    parsed_args[:nworkers] = nworkers
    niterations::Int = parsed_args[:niterations]
    niterations > 0 || throw(DomainError(niterations, "The number of iterations must be non-negative"))
    nwait::Int = isnothing(parsed_args[:nwait]) ? nworkers : parsed_args[:nwait]    
    0 < nwait <= nworkers || throw(DomainError(nwait, "nwait must be in [1, nworkers]"))    
    timeout::Float64 = parsed_args[:timeout]

    # communication setup
    pool = AsyncPool(nworkers)    
    recvbuf = zeros(UInt8, nworkers*parsed_args[:nbytes])
    sendbuf = zeros(UInt8, parsed_args[:nbytes])
    irecvbuf = similar(recvbuf)    
    isendbuf = similar(sendbuf, nworkers*length(sendbuf))
    
    # views into recvbuf for each worker
    n = div(length(recvbuf), nworkers)
    recvbufs = [view(recvbuf, (i-1)*n+1:i*n) for i in 1:nworkers]
    tbufs = [reinterpret(Float64, view(buf, 1:8)) for buf in recvbufs]

    # store which workers responded in each iteration
    worker_repochs = zeros(Int, nworkers, niterations) # receive epoch for each worker and iteration
    worker_latency = zeros(nworkers, niterations) # latency of individual workers
    worker_compute_latency = zeros(nworkers, niterations) # compute latency recorded by the worker
    timestamps = zeros(UInt64, niterations)
    latency = zeros(niterations)

    # dummy iteration to force compilation
    timestamps[1] = time_ns()
    latency[1] = @elapsed begin
        repochs = asyncmap!(pool, sendbuf, recvbuf, isendbuf, irecvbuf, comm, nwait=nworkers, tag=data_tag)
    end
    worker_repochs[:, 1] .= repochs
    worker_latency[:, 1] .= pool.latency
    if timeout > 0
        sleep(timeout)
    end    

    # main loop
    for i in 1:parsed_args[:niterations]
        timestamps[i] = time_ns()
        latency[i] = @elapsed begin
            repochs = asyncmap!(pool, sendbuf, recvbuf, isendbuf, irecvbuf, comm, nwait=nwait, tag=data_tag)
        end
        worker_repochs[:, i] .= repochs
        worker_latency[:, i] .= pool.latency
        for j in 1:nworkers
            worker_compute_latency[j, i] = tbufs[j][1]
        end
        if timeout > 0
            sleep(timeout)
        end
    end

    # write statistics to file
    h5open(parsed_args[:outputfile], "w") do fid
        for (key, val) in parsed_args
            if isnothing(val)
                continue
            end
            fid["parameters/$key"] = val
        end        
        fid["worker_repochs"] = worker_repochs
        fid["worker_latency"] = worker_latency
        fid["worker_compute_latency"] = worker_compute_latency
        fid["timestamps"] = timestamps
        fid["latency"] = latency
    end
    return
end

if isroot
    coordinator_main()
else
    # try/catch to eliminate stacktrace
    try
        worker_main()
    catch e
        printstyled(stderr,"ERROR: ", bold=true, color=:red)
        printstyled(stderr,sprint(showerror,e), color=:light_red)
        println(stderr)
    end
end
MPI.Barrier(comm)