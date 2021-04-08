function netcode_timeseries(dfo, jobid)
    dfo = filter(:jobid => x -> x == jobid, dfo)
    dfo.worker_communication_latency = dfo.worker_latency .- dfo.worker_compute_latency
    sort!(dfo, [:worker_index, :iteration])
    t0 = dfo.timestamp[1]
    for worker_index in unique(dfo.worker_index)
        dfi = filter(:worker_index => x -> x == worker_index, dfo)
        plt.plot(dfi.worker_communication_latency, ".", label="Worker $worker_index")
    end
    plt.grid()
    plt.xlabel("Iteration index")
    plt.ylabel("Communication latency [s]")
    return
end

function netcode_cdfs(dfo, jobid; ccdf=false)
    dfo = filter(:jobid => x -> x == jobid, dfo)
    dfo.worker_communication_latency = dfo.worker_latency .- dfo.worker_compute_latency
    sort!(dfo, [:worker_index, :iteration])
    t0 = dfo.timestamp[1]
    plt.figure()
    for worker_index in unique(dfo.worker_index)
        dfi = filter(:worker_index => x -> x == worker_index, dfo)
        xs = sort(dfi.worker_communication_latency)
        ys = range(0, 1, length=length(xs))
        if ccdf
            ys .= 1 .- ys
        end
        plt.plot(xs, ys, label="Worker $worker_index")
    end
    plt.grid()
    plt.xlabel("Iteration index")
    plt.ylabel("Communication latency [s]")
    if ccdf
        plt.yscale("log")
    end
    return
end