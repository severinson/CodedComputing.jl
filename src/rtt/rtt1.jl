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

function shutdown(pool::StragglerPool)
    for i in pool.ranks
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
    pool = StragglerPool(nworkers)
    sendbuf = Vector{Float64}(undef, nelems)
    isendbuf = Vector{Float64}(undef, nworkers*length(sendbuf))
    recvbuf = Vector{Float64}(undef, nworkers*nelems)
    irecvbuf = copy(recvbuf)

    # warmup (wait for all workrs)
    print("warming up\n")
    epoch = 1
    kmap!(sendbuf, recvbuf, isendbuf, irecvbuf, nworkers, epoch, pool, comm; tag=data_tag)

    # output file
    f = open(filename, "a")
    write(f, "nworkers, nwait, nMB_m2w, nMB_w2m, rtt\n")
    nMB_m2w = nelems * 64/8 / 1e6 # MB sent from master to workers
    nMB_w2m = nelems * 64/8 / 1e6 # MB sent from workers to master

    # collect rtt samples
    print("starting test\n")
    for k in 1:nworkers
        for _ in 1:nsamples
            epoch += 1
            rtt = @elapsed kmap!(sendbuf, recvbuf, isendbuf, irecvbuf, k, epoch, pool, comm; tag=data_tag)
            write(f, "$nworkers, $k, $nMB_m2w, $nMB_w2m, $rtt\n")
        end
    end

    print("tests done\n")
    close(f)
    shutdown(pool)
end

function worker_main()
    recvbuf = Vector{Float64}(undef, nelems)
    sendbuf = Vector{Float64}(undef, nelems)
    crreq = MPI.Irecv!(zeros(1), root, control_tag, comm)
    while true
        rreq = MPI.Irecv!(recvbuf, root, data_tag, comm)
        index, _ = MPI.Waitany!([crreq, rreq])
        if index == 1 # exit message on control channel
            break
        end
        sendbuf .= recvbuf
        sreq = MPI.Isend(sendbuf, root, data_tag, comm)
    end
end

if isroot()
    root_main()
else
    worker_main()
end
MPI.Barrier(comm)