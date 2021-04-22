"""

Plot order statistics latency for a given computational load.
"""
function plot_orderstats(df; nworkers=nothing, worker_flops=2.27e7)
    if !isnothing(nworkers)
        df = filter(:nworkers => (x)->x==nworkers, df)
    end
    if !isnothing(worker_flops)
        df = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), df)
    end
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
        if size(dfi, 1) == 0
            continue
        end
        for worker_flops in sort!(unique(dfi.worker_flops))
            println((nworkers, worker_flops))
            dfj = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), dfi)
            if size(dfj, 1) == 0
                continue
            end

            # empirical latency        
            for nwait in 1:nworkers
                # dfk = filter(:nwait => (x)->x>=nwait, dfj)
                ys[nwait] = mean(dfj[:, latency_columns[nwait]])
            end
            xs = 1:nworkers
            ys = view(ys, 1:nworkers)
            plt.plot(xs, ys, "-o", label="Nn: $nworkers, c: $worker_flops")
            write_table(xs, ys, "./results/orderstats_$(nworkers)_$(worker_flops).csv")

            # latency predicted by the degree-3 model (local)
            p, _ = fit_polynomial(xs, ys, 3)            
            ys = p.(1:nworkers)
            plt.plot(xs, ys, "c--")
            write_table(xs, ys, "./results/orderstats_deg3l_$(nworkers)_$(worker_flops).csv")

            # latency predicted by the degree-3 model (global)
            ys = predict_latency.(1:nworkers, worker_flops, nworkers)
            plt.plot(xs, ys, "k--")
            write_table(xs, ys, "./results/orderstats_deg3_$(nworkers)_$(worker_flops).csv")

            # latency predicted by the shifted exponential model
            ys = predict_latency_shiftexp.(1:nworkers, worker_flops, nworkers)
            plt.plot(xs, ys, "m--")
            write_table(xs, ys, "./results/orderstats_shiftexp_$(nworkers)_$(worker_flops).csv")
        end
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Order")        
    plt.ylabel("Latency [s]")
    plt.tight_layout()
    return
end

### degree-3 polynomial latency model (fitted globally to all samples)

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

Return the degree-3 model coefficients, computed using `fit_deg3_model`.
"""
function deg3_coeffs(type="c5xlarge")
    if type == "c5xlarge"
        b1 = -0.0005487338276092924
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

Return the latency predicted by the degree-3 model.
"""
function predict_latency(nwait, nflops, nworkers; type="c5xlarge")
    b1, c1, d1, e1, b2, c2, d2, e2 = deg3_coeffs(type)
    rv = b1 + b2*nflops
    rv += c1*nwait + c2*nflops*nwait/nworkers
    rv += d1*nwait^2 + d2*nflops*(nwait/nworkers)^2
    rv += e1*nwait^3 + e2*nflops*(nwait/nworkers)^3
    rv
end

### local degree-3 polynomial model (fitted to unique combinations of nworkers and worker_flops)

"""

Return a DataFrame, where each row corresponds to a unique combination of `nworkers` and
`worker_flops`, and the latency is an average computed over all experiments for that
combination.
"""
function mean_latency_df(df)
    df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    fs = [latency_columns[i] => mean => latency_columns[i] for i in 1:maxworkers]
    combine(groupby(df, [:worker_flops, :nworkers]), fs...)
end

"""

Fit a degree-3 polynomial to the latency of the `w`-th fastest worker as a function of `w` for each
unique combination of `nworkers` and `worker_flops`.
"""
function fit_local_deg3_model(dfm)
    maxworkers = maximum(dfm.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]    
    rv = DataFrame()    
    row = Dict{String,Any}()
    buffer = zeros(maxworkers)
    for i in 1:size(dfm, 1)
        nworkers = dfm[i, "nworkers"]
        row["nworkers"] = nworkers
        row["worker_flops"] = dfm[i, "worker_flops"]
        for j in 1:nworkers
            buffer[j] = dfm[i, latency_columns[j]]
        end
        _, coeffs = fit_polynomial(1:nworkers, view(buffer, 1:nworkers), 3)
        for j in 1:4
            row["x$j"] = coeffs[j]
        end
        push!(rv, row, cols=:union)
    end
    rv
end

function plot_deg3_model(df3)
    plt.figure()
    for (i, col) in enumerate([:x1, :x2, :x3, :x4])
        plt.subplot(2, 2, i)
        for nworkers in sort!(unique(df3.nworkers))
            dfi = filter(:nworkers => (x)->x==nworkers, df3)
            xs = dfi[:worker_flops] ./ nworkers^(i-1)
            ys = dfi[:, col]
            if i == 3
                ys .*= -1
            end
            plt.plot(xs, ys, ".", label="$nworkers workers")
            write_table(xs, ys, "./results/deg3_$(col)_$(nworkers).csv")
        end
        if i == 3
            plt.ylabel("-$col")
        else
            plt.ylabel(col)
        end
        plt.xlabel("c / nworkers^$(i-1)")
        plt.grid()
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()        
    end
    return
end

### shifted exponential latency model

get_shift(nflops) = 0.004285760641619949 + 7.473244861623717e-9nflops
get_scale(nflops) = -0.00020143027101769708 + 4.345057865353295e-10nflops

"""

Fit a shifted exponential latency model for a specific value of worker_flops.
"""
function fit_shiftexp_model(df, worker_flops)
    df = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), df)
    df = filter(:nreplicas => (x)->x==1, df)
    if size(df, 1) == 0 || minimum(df.nwait) != 1
        return NaN, NaN
    end

    # get the shift from waiting for 1 worker
    shift = quantile(df[df.nwait .== 1, :latency], 0.01)
    ts = df.latency .- shift

    # get the scale from waiting for all workers
    β = 0.0
    for nworkers in unique(df.nworkers)
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        nwait = nworkers
        ts = dfi[dfi.nwait .== nwait, :latency] .- shift
        # σ = var(ts)
        # β1 = sqrt(σ / sum(1/i^2 for i in (nworkers-nwait+1):nworkers))        
        μ = mean(ts)
        βi = μ / sum(1/i for i in (nworkers-nwait+1):nworkers)
        β += βi * size(dfi, 1) / size(df, 1)
    end
    return shift, β
end

"""

Fit a shifted exponential model to the data.
"""
function fit_shiftexp_model(df)
    ws = sort!(unique(df.worker_flops))
    models = [fit_shiftexp_model(df, w) for w in ws]
    shifts = [m[1] for m in models]
    scales = [m[2] for m in models]

    # filter out nans
    mask = findall(.!isnan, scales)
    ws = ws[mask]
    shifts = shifts[mask]
    scales = scales[mask]

    plt.figure()
    plt.plot(ws, shifts, "o")

    poly = Polynomials.fit(ws, shifts, 1)
    println("Shift polynomial coefficients: $(poly.coeffs)")
    ts = range(0, maximum(df.worker_flops), length=100)
    plt.plot(ts, poly.(ts))

    plt.grid()
    plt.xlabel("w")    
    plt.ylabel("shift")

    plt.figure()
    plt.plot(ws, scales, "o")

    poly = Polynomials.fit(ws, scales, 1)
    println("Scale polynomial coefficients: $(poly.coeffs)")
    ts = range(0, maximum(df.worker_flops), length=100)
    plt.plot(ts, poly.(ts))    

    plt.grid()
    plt.xlabel("w")    
    plt.ylabel("scale")    

    return
end

"""

Return the latency predicted by the shifted exponential model.
"""
function predict_latency_shiftexp(nwait, nflops, nworkers; type="c5xlarge")
    type == "c5xlarge" || error("only parameters for c5xlarge are available")
    shift, scale = get_shift(nflops), get_scale(nflops)
    if shift < 0 || scale < 0
        return NaN
    end
    shift + mean(ExponentialOrder(scale, nworkers, nwait))
end

### predicted and empirical latency

"""

Plot latency as a function of `nworkers` for a fixed total number of flops across all workers per 
iteration, denoted by `c0`.
"""
function plot_predictions(c0=1.6362946777247114e9; df=nothing)
    if !isnothing(df)
        df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    end
    nworkers = 1:200
    c = c0 ./ nworkers
    plt.figure()
    for ϕ in [0.5, 1.0]
        nwait = ϕ.*nworkers
        xs = nworkers
        ys = predict_latency.(nwait, c, nworkers)
        plt.plot(xs, ys, label="$(ϕ) (analytic)")
        description = "$(round(c0, sigdigits=3))_$(round(ϕ, sigdigits=3))"
        write_table(xs, ys, "./results/pred_"*description*".csv")
        if !isnothing(df)
            dfi = filter([:worker_flops, :nworkers] => (x, y)->isapprox(x*y, c0, rtol=1e-2), df)
            xs = zeros(Int, 0)
            ys = zeros(0)
            for nworkers in sort!(unique(dfi.nworkers))
                dfj = filter(:nworkers => (x)->x==nworkers, dfi)
                if size(dfj, 1) == 0
                    continue
                end
                push!(xs, nworkers)
                push!(ys, mean(dfj[:, "latency_worker_$(round(Int, ϕ*nworkers))"]))
            end
            plt.plot(xs, ys, "o", label="$ϕ (empirical")
            write_table(xs, ys, "./results/emp_"*description*".csv")            
        end      
    end
    plt.yscale("log")
    plt.legend()
    plt.xlabel("nworkers")
    plt.ylabel("Latency [s]")
    plt.grid()
    return
end

### probability of stragglers remaining stragglers

"""

Return a DataFrame, for which each row corresponds to a `jobid` of the input `df`, containing the
following probabilities for each worker:

- The probability of not being among the `w` fastest workers, given that it was among the `w` 
  fastest in the previous iteration, denoted by `pr12`
- The probability of being among the `w` fastest workers, given that it was not among the `w` 
  fastest in the previous iteration, denoted by `pr21`

"""
function straggler_transition_probabilities(df, w)
    df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    rv = DataFrame()
    row = Dict{String,Any}()
    maxworkers = maximum(df.nworkers)
    platencies = zeros(maxworkers) # latency in prev. iteration
    latencies = zeros(maxworkers) # latency in this iteration
    pr12s = zeros(maxworkers)
    pr21s = zeros(maxworkers)
    s1s = zeros(Int, maxworkers)
    s2s = zeros(Int, maxworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    for jobid in unique(df.jobid)
        dfi = filter(:jobid => (x)->x==jobid, df)
        sort!(dfi, :iteration)
        nworkers = dfi[1, :nworkers]
        row["jobid"] = jobid
        row["nworkers"] = nworkers
        row["nwait"] = dfi[1, :nwait]
        row["worker_flops"] = dfi[1, :worker_flops]

        for j in 1:nworkers
            platencies[j] = dfi[1, latency_columns[j]]            
        end     
        sort!(view(platencies, 1:nworkers))
        pr12s .= 0
        pr21s .= 0
        s1s .= 0
        s2s .= 0
        for i in 2:size(dfi, 1)
            if w > nworkers
                continue
            end
            for j in 1:nworkers
                latencies[j] = dfi[i, latency_columns[j]]
            end
            sort!(view(latencies, 1:nworkers))
            for j in 1:nworkers
                sp = dfi[i-1, latency_columns[j]] <= platencies[w] # among the w fastest in the prev. iteration
                s = dfi[i, latency_columns[j]] <= latencies[w] # among the w fastest in this iteration
                if sp
                    s1s[j] += 1
                    if s == false # state 1 => state 2 (became a straggler)
                        pr12s[j] += 1
                    end
                else
                    s2s[j] += 1
                    if s # state 2 => state 1 (no longer a straggler)
                        pr21s[j] += 1
                    end
                end
            end
            platencies .= latencies
        end
        for j in 1:nworkers            
            row["worker_index"] = j
            row["pr12"] = s1s[j] == 0 ? missing : pr12s[j] / (s1s[j] + 1)
            row["pr21"] = s2s[j] == 0 ? missing : pr21s[j] / (s2s[j] + 1)
            push!(rv, row, cols=:union)
        end
    end
    rv
end

"""

Plot the empirical distribution of the probability that a worker that is not among the `ϕ*nworkers`
fastest workers in the previous iteration is also not among the `ϕ*nworkers` fastest workers in 
this iteration.
"""
function plot_transition_probability(df; ϕ=1/2, worker_flops=1.14e7)
    df = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), df)
    1.14e7
    1.52e7
    plt.figure()
    for nworkers in sort!(unique(df.nworkers))
        w = round(Int, nworkers*ϕ)
        dfi = filter(:nworkers => (x)->x==nworkers, df)
        dfp = straggler_transition_probabilities(dfi, w)
        pr22 = 1 .- skipmissing(dfp.pr21)
        sort!(pr22)
        xs = range(0, 1, length=length(pr22))
        plt.plot(xs, pr22, label="$nworkers workers")
    end
    
    # plot iid model
    xs = [0, 0, 1, 1]
    ys = [0, 1-ϕ, 1-ϕ, 1]
    plt.plot(xs, ys, "k-", label="iid model")

    plt.legend()
    plt.grid()
    plt.xlabel("Fraction of workers")
    plt.ylabel("Pr. still a straggler")
    return
end