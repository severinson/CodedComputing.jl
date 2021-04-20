"""

Plot order statistics latency for a given computational load.
"""
function plot_orderstats(df; worker_flops=2.27e7)
    df = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), df)
    df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    if size(df, 1) == 0
        println("No rows match constraints")
        return
    end
    println("worker_flops:\t$(unique(df.worker_flops))")
    println("nbytes:\t$(unique(df.nbytes))\n")
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    ys = zeros(maxworkers)
    plt.figure()
    for nworkers in sort!(unique(df.nworkers))
        dfi = filter(:nworkers => (x)->x==nworkers, df)
        for nwait in 1:nworkers
            dfj = filter(:nwait => (x)->x>=nwait, dfi)
            ys[nwait] = mean(dfj[:, latency_columns[nwait]])
        end
        plt.plot(1:nworkers, view(ys, 1:nworkers), "-o", label="Nn: $nworkers")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Order")        
    plt.ylabel("Latency [s]")
    plt.tight_layout()
    return
end

"""

Fit the degree-3 model to the latency data.
"""
function fit_deg3_model(df)
    A = zeros(sum(df.nwait), 8)
    A[:, 1] .= 1
    y = zeros(size(A, 1))
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    k = 1
    for i in 1:size(df, 1)
        for j in 1:df[i, :nwait]
            A[k, 2] = j
            A[k, 3] = j^2
            A[k, 4] = j^3
            A[k, 5] = df[i, :worker_flops]
            A[k, 6] = df[i, :worker_flops] * j / df[i, :nworkers]
            A[k, 7] = df[i, :worker_flops] * (j / df[i, :nworkers])^2
            A[k, 8] = df[i, :worker_flops] * (j / df[i, :nworkers])^3
            y[k] = df[i, latency_columns[j]]
            k += 1
        end
    end
    x = A\y
    for (i, label) in enumerate(["b1", "c1", "d1", "e1", "b2", "c2", "d2", "e2"])
        println("$label = $(x[i])")
    end    
    x
end

"""

Return the degree-3 model coefficients.
"""
function deg3_coeffs(type="c5xlarge")
    if type == "c5xlarge"
        # b1 = -0.0005487338276092924
        b1 = 0
        c1 = 0.00011666153003402824
        d1 = -2.200065092782715e-6
        e1 = 1.3139560334678954e-8
        b2 = 7.632075760960183e-9
        c2 = 2.1903320927807077e-9
        d2 = -4.525831193535335e-9
        e2 = 4.336744075595763e-9
        return b1, c1, d1, e1, b2, c2, d2, e2
    elseif type == "t3large"
        b1 = -0.0012538429018191268
        c1 = 5.688267095613402e-5
        d1 = 1.8724136277744778e-6
        e1 = -1.2889725208620691e-8
        b2 = 8.140573448894689e-9
        c2 = 5.388607340950452e-9
        d2 = -1.1648036394321019e-8
        e2 = 7.880211300623262e-9
        return b1, c1, d1, e1, b2, c2, d2, e2
    end    
    error("no instance type $type")
end

"""

Return the latency predicted by the degree-3 model, where `c` is the number of flops per worker and iteration.
"""
function predict_latency(c, nwait, nworkers; type="c5xlarge")
    b1, c1, d1, e1, b2, c2, d2, e2 = deg3_coeffs(type)
    rv = b1 + b2*c
    rv += c1*nwait + c2*c*nwait/nworkers
    rv += d1*nwait^2 + d2*c*(nwait/nworkers)^2
    rv += e1*nwait^3 + e2*c*(nwait/nworkers)^3
    rv
end