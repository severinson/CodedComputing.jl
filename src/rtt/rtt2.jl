using MPI
using MPIStragglers

MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0
const rank = MPI.Comm_rank(comm)
# const nworkers = MPI.Comm_size(comm) - 1
isroot() = MPI.Comm_rank(comm) == root
const data_tag = 0
const control_tag = 1
const nelems = 100 # number of 64-bit floats sent in each direction per iteration and worker

function shutdown()
    for i in 1:(MPI.Comm_size(comm)-1)
        MPI.Isend(zeros(1), i, control_tag, comm)
    end
end

function root_main()    

    # parse command-line-arguments
    nworkers = MPI.Comm_size(comm) - 1
    println((nworkers, rank, root))
    print("starting with arguments $ARGS\n")
    nsamples = parse(Int, ARGS[1])
    0 < nsamples || throw(ArgumentError("The number of samples must be positive"))
    filename = ARGS[2]

    # setup
    sendbuf = Vector{Float64}(undef, nelems)
    recvbuf = Vector{Float64}(undef, (nworkers+1)*nelems)

    # warmup
    print("warming up\n")
    MPI.Bcast!(sendbuf, root, comm)
    MPI.Gather!(sendbuf, recvbuf, root, comm)

    # output file
    f = open(filename, "a")
    write(f, "nworkers, nMB_m2w, nMB_w2m, rtt\n")
    nMB_m2w = nelems * 64/8 / 1e6 # MB sent from master to workers
    nMB_w2m = nelems * 64/8 / 1e6 # MB sent from workers to master

    # collect rtt samples
    print("starting test\n")
    for _ in 1:nsamples
        rtt = @elapsed begin
            MPI.Bcast!(sendbuf, root, comm)
            MPI.Gather!(sendbuf, recvbuf, root, comm)
        end
        write(f, "$nworkers, $nMB_m2w, $nMB_w2m, $rtt\n")
    end

    print("tests done\n")
    close(f)
    shutdown()
end

function worker_main()
    recvbuf = Vector{Float64}(undef, nelems)
    sendbuf = Vector{Float64}(undef, nelems)
    crreq = MPI.Irecv!(zeros(1), root, control_tag, comm)
    while true

        # check for exit message on control channel
        flag, _ = MPI.Test!(crreq)
        if flag
            break
        end

        # send/receive
        MPI.Bcast!(recvbuf, root, comm)
        sendbuf .= recvbuf
        MPI.Gather!(sendbuf, nothing, root, comm)
    end
end

if isroot()
    root_main()
else
    worker_main()
end
MPI.Barrier(comm)