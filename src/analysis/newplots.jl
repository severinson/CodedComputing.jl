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
    df = filter(:iteration => (x)->x>1, df)
    if size(df, 1) == 0
        println("No rows match constraints")
        return
    end
    println("worker_flops:\t$(unique(df.worker_flops))")
    println("nbytes:\t$(unique(df.nbytes))\n")
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    orderstats = zeros(maxworkers)
    plt.figure()
    for nworkers in sort!(unique(df.nworkers))
        dfi = filter(:nworkers => (x)->x==nworkers, df)
        if size(dfi, 1) == 0
            continue
        end
        for worker_flops in [1.13e7, 1.51e7, 2.27e7, 4.54e7]
            # for worker_flops in sort!(unique(dfi.worker_flops))
            println((nworkers, worker_flops))
            dfj = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), dfi)
            if size(dfj, 1) == 0
                continue
            end

            # empirical latency        
            for nwait in 1:nworkers
                # dfk = filter(:nwait => (x)->x>=nwait, dfj)
                orderstats[nwait] = mean(dfj[:, latency_columns[nwait]])
            end
            xs = 1:nworkers
            ys = view(orderstats, 1:nworkers)
            plt.plot(xs, ys, "-o", label="Nn: $nworkers, c: $worker_flops")
            write_table(xs, ys, "./results/orderstats_$(nworkers)_$(worker_flops).csv")

            println("Acuteness: $(ys[end] / ys[1])")

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

            # latency predicted by the gamma model
            ys = predict_latency_gamma(worker_flops, nworkers)
            plt.plot(xs, ys, "k-")
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

    # coefficients of global model
    nflops1 = 10 .^ range(log10(minimum(df3.worker_flops)), log10(maximum(df3.worker_flops)), length=100)
    nflops2 = 10 .^ range(log10(minimum(df3.worker_flops./df3.nworkers)), log10(maximum(df3.worker_flops./df3.nworkers)), length=100)
    nflops3 = 10 .^ range(log10(minimum(df3.worker_flops./df3.nworkers.^2)), log10(maximum(df3.worker_flops./df3.nworkers.^2)), length=100)
    nflops4 = 10 .^ range(log10(minimum(df3.worker_flops./df3.nworkers.^3)), log10(maximum(df3.worker_flops./df3.nworkers.^3)), length=100)
    b1, c1, d1, e1, b2, c2, d2, e2 = deg3_coeffs("c5xlarge")
    α1 = b1 .+ b2.*nflops1
    α2 = c1 .+ c2.*nflops2
    α3 = -(d1 .+ d2.*nflops3)
    α4 = e1 .+ e2.*nflops4
    coefficients = [α1, α2, α3, α4]
    nflops_all = [nflops1, nflops2, nflops3, nflops4]

    plt.figure()
    for (i, col) in enumerate([:x1, :x2, :x3, :x4])
        plt.subplot(2, 2, i)

        # coefficients of local model
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

        xs = nflops_all[i]
        ys = coefficients[i]
        plt.plot(xs, ys, "k-")
        write_table(xs, ys, "./results/deg3_global_$(col).csv")

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
function plot_predictions(c0=1.6362946777247114e9; df=nothing, maxworkers=200)
    if !isnothing(df)
        df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    end
    nworkers = 1:maxworkers
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
function plot_transition_probability(df; ϕ=1/2, worker_flops=1.51e7)
    df = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), df)
    plt.figure()
    for nworkers in sort!(unique(df.nworkers))
        w = round(Int, nworkers*ϕ)
        dfi = filter(:nworkers => (x)->x==nworkers, df)
        dfp = straggler_transition_probabilities(dfi, w)
        pr22 = 1 .- skipmissing(dfp.pr21)
        sort!(pr22)
        xs = range(0, 1, length=length(pr22))
        write_table(xs, pr22, "./results/tp_$(nworkers)_$(ϕ)_$(round(worker_flops, sigdigits=3)).csv")
        plt.plot(xs, pr22, label="$nworkers workers")
    end
    
    # plot iid model
    xs = [0, 0, 1, 1]
    ys = [0, 1-ϕ, 1-ϕ, 1]
    write_table(xs, ys, "./results/tp_iid_$(ϕ).csv")
    plt.plot(xs, ys, "k-", label="iid model")

    plt.legend()
    plt.grid()
    plt.xlabel("Fraction of workers")
    plt.ylabel("Pr. still a straggler")
    return
end

"""

Plot the average latency of each worker computed over the first half of jobs vs. the average 
latency computed over the second half of jobs. Use this plot to verify that the workload is 
balanced across workers; the latency of a given worker will be correlated between jobs if the
workload is unbalanced.
"""
function plot_latency_balance(df; nsubpartitions=1)
    df = filter(:nsubpartitions => (x)->x==nsubpartitions, df)
    df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    plt.figure()
    for nworkers in sort!(unique(df.nworkers))
        xs = 1:nworkers
        dfi = filter(:nworkers => (x)->x==nworkers, df)
        j = round(Int, size(dfi, 1)/2)
        l1 = [mean(skipmissing(dfi[1:j, "latency_worker_$i"])) for i in 1:nworkers]
        l2 = [mean(skipmissing(dfi[(j+1):end, "latency_worker_$i"])) for i in 1:nworkers]
        plt.plot(l1, l2, "o", label="$nworkers workers")
    end
    plt.xlabel("Avg. latency (first half of jobs)")
    plt.ylabel("Avg. latency (second half of jobs)")
    plt.grid()
    plt.legend()
    return
end

### latency timeseries plots

"""

Plot the iteration latency of workers with indices in `workers` of job `jobid`.
"""
function plot_timeseries(df; jobid=rand(df.jobid), workers=[1, 2])
    println("jobid: $jobid")
    df = filter(:jobid => (x)->x==jobid, df)
    plt.figure()
    for worker in workers
        xs = df.iteration
        ys = df[:, "latency_worker_$worker"]
        plt.plot(xs, ys, label="Worker $worker")
        write_table(xs, ys, "./results/timeseries_$(jobid)_$(worker).csv")
    end
    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Per-worker iteration latency [s]")
    return
end

### convergence plots

"""

Plot the rate of convergence over time for DSAG, SAG, SGD, and coded computing. Let 
`latency=empirical` to plot against empirical latency, or let `latency=c5xlarge` to plot against 
latency computed by the model, fitted to traces recorded on `c5xlarge` instances.
"""
function plot_convergence(df, nworkers, opt=maximum(skipmissing(df.mse)); latency="empirical")
    df = filter(:nworkers => (x)->x==nworkers, df)
    df = filter(:nreplicas => (x)->x==1, df)
    df = filter(:mse => (x)->!ismissing(x), df)
    println("nworkers: $nworkers, opt: $opt")

    # parameters are recorded as a tuple (nwait, nsubpartitions, stepsize)
    if nworkers == 36

        # # varying npartitions
        # nwait = 3
        # params = [
        #     # (nwait, 10, 0.9),            
        #     # (nwait, 40, 0.9),
        #     # (nwait, 80, 0.9),
        #     (nwait, 160, 0.9),
        # ]

        # varying nwait      
        nsubpartitions = 160
        params = [
            (1, nsubpartitions, 0.9),
            (3, nsubpartitions, 0.9),            
            (6, nsubpartitions, 0.9),                        
            (9, nsubpartitions, 0.9),            
        ]
    elseif nworkers == 72
        nsubpartitions = 160
        params = [
            (1, nsubpartitions, 0.9),            
            (3, nsubpartitions, 0.9),            
            (6, nsubpartitions, 0.9),                        
            (9, nsubpartitions, 0.9),
        ]
        # nwait = 9
        # params = [
        #     (nwait, 120, 0.9),            
        #     (nwait, 160, 0.9),            
        #     # (nwait, nsubpartitions, 0.9),                        
        #     # (nwait, nsubpartitions, 0.9),
        # ]   
    elseif nworkers == 108
        nsubpartitions = 160
        params = [
            (1, nsubpartitions, 0.9),            
            (3, nsubpartitions, 0.9),            
            (6, nsubpartitions, 0.9),                        
            (9, nsubpartitions, 0.9),
        ]      
        # nwait = 3
        # params = [
        #     (nwait, 120, 0.9),            
        #     (nwait, 160, 0.9),            
        #     (nwait, 240, 0.9),            
        #     (nwait, 320, 0.9),            
        #     (nwait, 640, 0.9),            
        # ]  
    else
        error("parameters not defined")
    end

    plt.figure()    

    for (nwait, nsubpartitions, stepsize) in params

        dfi = df
        dfi = dfi[dfi.nwait .== nwait, :]
        dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
        dfi = dfi[dfi.stepsize .== stepsize, :]
        println("nwait: $nwait, nsubpartitions: $nsubpartitions, stepsize: $stepsize")

        ### DSAG
        dfj = dfi
        dfj = dfj[dfj.variancereduced .== true, :]
        if nwait < nworkers # for nwait = nworkers, DSAG and SAG are the same
            dfj = dfj[dfj.nostale .== false, :]
        end
        # return dfj
        println("DSAG: $(length(unique(dfj.jobid))) jobs")
        if size(dfj, 1) > 0
            dfj = combine(groupby(dfj, :iteration), :mse => mean => :mse, :time => mean => :time)
            if latency == "empirical"
                println("Plotting DSAG with empirical latency")
            else
                dfj.time .= predict_latency(nwait, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration
                println("Plotting DSAG with model latency for $latency")
            end
            xs = dfj.time
            ys = opt.-dfj.mse
            plt.semilogy(xs, ys, ".-", label="DSAG w=$nwait, p=$nsubpartitions")
            filename = "./results/dsag_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"            
            write_table(xs, ys, filename)
        end
        println()
    end

    # Plot SAG
    # for nsubpartitions in sort!(unique(df.nsubpartitions))
    nsubpartitions = 160
    # for nsubpartitions in [80, 120, 160, 240, 320]
    stepsize = 0.9
    dfi = df
    dfi = dfi[dfi.nwait .== nworkers, :]
    dfi = dfi[dfi.variancereduced .== true, :]
    dfi = dfi[dfi.stepsize .== stepsize, :]    
    dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
    # dfi = dfi[dfi.nostale .== true, :]
    println("SAG p: $nsubpartitions, $(length(unique(dfi.jobid))) jobs")
    dfj = by(dfi, :iteration, :mse => mean => :mse, :time => mean => :time)
    sort!(dfj, :iteration)    
    if latency == "empirical"
        println("Plotting SAG with empirical latency")
    else
        dfj.time .= predict_latency(nworkers, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration
        println("Plotting SAG with model latency for $latency")
    end
    if size(dfj, 1) > 0
        xs = dfj.time
        ys = opt.-dfj.mse
        plt.semilogy(xs, ys, "o-", label="SAG p=$nsubpartitions")
        filename = "./results/sag_$(nworkers)_$(nsubpartitions)_$(stepsize).csv"
        write_table(xs, ys, filename)        
    end
    # end

    # Plot SGD
    nsubpartitions = 160
    stepsize = 0.9
    dfi = df
    dfi = dfi[dfi.nwait .== nworkers, :]
    dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
    dfi = dfi[dfi.variancereduced .== false, :]
    dfi = dfi[dfi.stepsize .== stepsize, :]
    println("SGD p: $nsubpartitions, $(length(unique(dfi.jobid))) jobs")
    dfj = by(dfi, :iteration, :mse => mean => :mse, :time => mean => :time)
    sort!(dfj, :iteration)
    if latency == "empirical"
        println("Plotting SGD with empirical latency")
    else
        dfj.time .= predict_latency(nworkers, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration
        println("Plotting SGD with model latency for $latency")
    end    
    if size(dfj, 1) > 0
        xs = dfj.time
        ys = opt.-dfj.mse
        plt.semilogy(xs, ys, "c^-", label="SGD p=$nsubpartitions")
        filename = "./results/sgd_$(nworkers)_$(nsubpartitions)_$(stepsize).csv"
        write_table(xs, ys, filename)        
    end

    # Plot GD
    stepsize = 1.0
    dfi = df
    dfi = dfi[dfi.nwait .== nworkers, :]
    dfi = dfi[dfi.nsubpartitions .== 1, :]
    dfi = dfi[dfi.variancereduced .== false, :]
    dfi = dfi[dfi.stepsize .== stepsize, :]
    println("GD $(length(unique(dfi.jobid))) jobs")
    dfj = by(dfi, :iteration, :mse => mean => :mse, :time => mean => :time)
    if latency == "empirical"
        println("Plotting GD with empirical latency")
    else
        dfj.time .= predict_latency(nworkers, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration
        println("Plotting GD with model latency for $latency")
    end    
    if size(dfj, 1) > 0
        xs = dfj.time
        ys = opt.-dfj.mse
        plt.semilogy(xs, ys, "ms-", label="GD")
        filename = "./results/gd_$(nworkers)_$(stepsize).csv"
        write_table(xs, ys, filename)
    end

    # plot coded computing
    r = 2 # replication factor
    Nw = 1 # number of workers to wait for
    samp = 1 # workload up-scaling

    # get the average error per iteration of GD
    dfi = df
    dfi = dfi[dfi.nsubpartitions .== 1, :]
    dfi = dfi[dfi.nwait .== nworkers, :]
    dfi = dfi[dfi.stepsize .== 1, :]
    dfi = dfi[dfi.variancereduced .== false, :]
    dfi = dfi[dfi.nostale .== false, :]
    dfj = combine(groupby(dfi, :iteration), :mse => mean => :mse)
    sort!(dfj, :iteration)
    ys = opt .- dfj.mse

    # compute the iteration time for a scheme with a factor r replication
    @assert length(unique(dfi.worker_flops)) == 1
    worker_flops = r*mean(dfi.worker_flops)
    t_iter = predict_latency(Nw, worker_flops, nworkers)
    xs = t_iter .* dfj.iteration

    # make the plot
    plt.semilogy(xs, ys, "--k", label="Bound r: $r, Nw: $Nw")
    # write_table(xs, ys, "./data/bound_$(nworkers)_$(Nw)_$(r).csv")
    filename = "./results/bound_$(nworkers)_$(stepsize).csv"
    write_table(xs, ys, filename)    

    plt.xlim(1e-2, 1e2)
    plt.xscale("log")
    plt.grid()
    plt.legend()    
    plt.xlabel("Time [s]")
    plt.ylabel("Explained Variance Sub-optimality Gap")
    return
end

### non-iid Gamma model

# Question: how do I evaluate the performance of my model?
# I mostly care about the average order statistics latency
# Hence, the way to evaluate the model is to compute the MSE.

function plot_worker_latency_distribution(df; jobid=1080, worker_indices=[10, 36])
    df = filter(:jobid => (x)->x==jobid, df)
    plt.figure()
    for i in worker_indices
        xs = sort(df[:, "latency_worker_$(i)"])
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys)
        d = Distributions.fit(ShiftedExponential, xs)
        plt.plot(xs, cdf.(d, xs), "k--", label="Worker $i")
    end
    plt.xlabel("Per-worker iteration latency [s]")
    plt.ylabel("CDF")
    worker_flops = df.worker_flops[1]
    plt.title("job $jobid")

    # # plot some generated distributions
    # worker_flops = df.worker_flops[1]
    # for _ in 1:3
    #     w = gamma_worker_distribution(worker_flops)
    #     xs = range(quantile(w, 0.001), quantile(w, 0.999), length=100)
    #     plt.plot(xs, cdf.(w, xs), "k-")
    # end

    plt.grid()
    return
end

"""

Fit a `Gamma` distribution to the latency of each worker for each job.
"""
function gamma_df(df; minsamples=100)
    df = filter([:nwait, :nworkers] => (x,y)->x==y, df)
    rv = DataFrame()
    row = Dict{String, Any}()
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]    
    for jobid in unique(df.jobid)
        dfi = filter(:jobid => (x)->x==jobid, df)
        nsamples = size(dfi, 1)
        if nsamples < minsamples
            continue
        end
        nworkers = dfi.nworkers[1]
        row["nworkers"] = nworkers
        row["worker_flops"] = dfi.worker_flops[1]
        row["nbytes"] = dfi.nbytes[1]
        row["nsamples"] = size(dfi, 1)
        row["jobid"] = jobid
        for i in 1:nworkers
            d = Distributions.fit(Gamma, float.(dfi[:, latency_columns[i]]))
            row["α"], row["θ"] = params(d)
            row["worker_index"] = i
            push!(rv, row, cols=:union)
        end
    end
    rv.mean = rv[:, "α"] .* rv[:, "θ"]
    rv.var = rv[:, "α"] .* rv[:, "θ"].^2
    rv
end

"""

Show how the avg. per-worker latency scales with nflops, and the distribution of the avg. 
per-worker latency.
"""
function plot_gamma_mean_distribution(dfg)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(dfg.worker_flops, dfg.mean, ".")
    plt.xlabel("nflops")
    plt.ylabel("Avg. per-worker latency")
    plt.grid()

    plt.subplot(1, 2, 2)
    for worker_flops in sort!(unique(dfg.worker_flops))
        dfi = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), dfg)
        xs = sort(dfi.mean)
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys)
        
        d = Distributions.fit(Gamma, xs)
        xs = range(quantile(d, 0.001), quantile(d, 0.999), length=100)
        plt.plot(xs, cdf.(d, xs), "k--")
    end
    # plt.yscale("log")
    plt.xlabel("Avg. per-worker latency")
    plt.ylabel("CDF")
    # plt.xscale("log")
    plt.grid()
    plt.tight_layout()
    return
end

"""

Fit a `ShiftedExponential` distribution to the avg. per-worker latency for each value of 
`worker_flops`.
"""
function gamma_mean_df(dfg; minsamples=400)
    rv = DataFrame()
    row = Dict{String, Any}()    
    for worker_flops in unique(dfg.worker_flops)
        dfi = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), dfg)
        nsamples = size(dfi, 1)
        if nsamples < minsamples
            continue
        end

        # Gamma model
        d = Distributions.fit(Gamma, dfi.mean)
        α, θ = params(d)
        row["α"] = α
        row["θ"] = θ
        row["mean_mean"], row["var_mean"] = α*θ, α*θ^2

        # # ShiftedExponential model
        # d = Distributions.fit(ShiftedExponential, dfi.mean)
        # s, θ = params(d)
        # row["s"] = s
        # row["θ"] = θ
        # row["mean_mean"], row["var_mean"] = s+θ, θ^2

        row["worker_flops"] = worker_flops
        row["nbytes"] = dfi.nbytes[1]
        row["nsamples"] = nsamples
        push!(rv, row, cols=:union)
    end
    rv
end

"""

Plot the parameters of the `ShiftedExponential` distribution fit to the mean latency vs. nflops.
"""
function plot_gamma_mean_df(dfgm)
    plt.figure()

    # shift vs. nflops
    plt.subplot(1, 2, 1)
    xs = dfgm.worker_flops
    ys = dfgm.mean_mean
    plt.plot(xs, ys, ".")
    plt.xlabel("workload [flops]")
    plt.ylabel("meta-mean (meta-shape * meta-scale)")
    plt.grid()

    ts = sort(xs)
    p, coeffs = fit_polynomial(xs, ys, 1)
    plt.plot(ts, p.(ts), "k--")
    println("shift coefficients: $coeffs")

    # plt.xscale("log")
    # plt.yscale("log")    

    # scale vs. nflops
    plt.subplot(1, 2, 2)
    ys = dfgm.θ
    plt.plot(xs, ys, ".")
    plt.xlabel("workload [flops]")
    plt.ylabel("meta-scale")
    plt.grid()
    
    p, coeffs = fit_polynomial(xs, ys, 1)
    plt.plot(ts, p.(ts), "k--")    
    println("scale coefficients: $coeffs")

    # plt.xscale("log")
    # plt.yscale("log")        

    plt.tight_layout()
end

function plot_gamma_var_distribution(dfg; nbins=100, minsamples=100)

    plt.figure()
    plt.subplot(1, 2, 1)
    xs = dfg.mean
    ys = dfg.θ # equal to dfg.var ./ dfg.mean    
    plt.plot(xs, ys, ".")
    plt.xlabel("avg. per-worker latency")
    plt.ylabel("per-worker latency var. / avg. per-worker latency")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()

    plt.subplot(1, 2, 2)
    xs = sort(ys)
    i = round(Int, 0.05*length(xs))
    xs = xs[i:end-i]
    ys = range(0, 1, length=length(xs))
    plt.semilogy(xs, 1 .- ys)

    d = Distributions.fit(LogNormal, xs)
    println(d)
    plt.plot(xs, 1 .- cdf.(d, xs), "k--")
    plt.ylabel("CCDF")
    plt.xlabel("per-worker latency var. / avg. per-worker latency")
    
    plt.xscale("log")
    plt.yscale("log")    
    plt.grid()
    plt.tight_layout()
    return
end


function plot_gamma_mean(dfgm)
    plt.figure()
    xs = dfgm.worker_flops
    ys = dfgm.mean_mean
    plt.plot(xs, ys, ".", label="Mean")
    p, coeffs = fit_polynomial(xs, ys, 1)
    println(coeffs)
    ts = sort(xs)
    plt.plot(ts, p.(ts), "k--")

    ys = sqrt.(dfgm.var_mean)
    plt.plot(xs, ys, ".", label="Var")
    p, coeffs = fit_polynomial(xs, ys, 1)
    println(coeffs)    
    plt.plot(ts, p.(ts), "k--")

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    return
end

function plot_gamma_var(dfg; nbins=20, minsamples=100)
    plt.figure()
    plt.plot(dfg.mean, dfg.var ./ dfg.mean, ".")


    # edges = 10.0.^range(log10(minimum(dfg.mean)), log10(maximum(dfg.mean)), length=nbins+1)
    # xs = zeros(0)
    # ys = zeros(0)
    # for i in 1:nbins
    #     dfi = filter(:mean => (x)->edges[i] <= x < edges[i+1], dfg)
    #     if size(dfi, 1) < minsamples
    #         continue
    #     end
    #     push!(xs, mean(dfi.mean))
    #     push!(ys, mean(dfi.var ./ dfi.mean))
    # end
    # plt.plot(xs, ys, "k-")

    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    return
end

"""

Plot the mean of the Gamma distrinution vs. flops.
"""
function plot_gamma_mean_vs_nflops(dfg)
    plt.figure()
    plt.plot(dfg.worker_flops, dfg.α.*dfg.θ, ".")
    plt.grid()
    return
end

"""

Generate a `Gamma` random variable for a worker with given per-worker workload
"""
function gamma_worker_distribution(worker_flops)

    # # ShiftedExponential model
    # # generate mean latency of the worker based on the per-worker workload
    # meta_shift = 0.0007499872578485828 + 7.778654414066668e-9worker_flops
    # meta_scale = -8.190460220896224e-5 + 5.58026508893015e-10worker_flops
    # d = ShiftedExponential(meta_shift, meta_scale)
    # mean = rand(d) # avg. latency of this worker

    # Gamma model
    meta_mean = 0.0006680826556396607 + 8.336680922959681e-9worker_flops
    meta_scale = -1.2720269369198488e-5 + 3.5940299549679294e-11worker_flops
    meta_shape = meta_mean / meta_scale
    d = Gamma(meta_shape, meta_scale)
    mean = rand(d)

    # scale parameter is selected at random, independently of the workload
    θ = rand(LogNormal(-9.642680987497839, 0.25878776103561885))
    
    # shape parameter is computed from the above
    α = mean / θ
    Gamma(α, θ)
end

"""

Compute all order statistics for `nworkers` workers, when the per-worker 
workload is `worker_flops`, via Monte Carlo sampling over `nsamples` samples.
"""
function predict_latency_gamma(worker_flops, nworkers; nsamples=1000)
    rv = zeros(nworkers)
    buffer = zeros(nworkers)
    for _ in 1:nsamples
        for i in 1:nworkers
            d = gamma_worker_distribution(worker_flops)
            buffer[i] = rand(d)
        end
        sort!(buffer)
        rv += buffer
    end
    rv ./= nsamples
end

"""

Compute the average latency of the `nwait`-th fastest out of `nworkers` 
workers, when the per-worker workload is `worker_flops`, via Monte Carlo
sampling over `nsamples` samples.
"""
function predict_latency_gamma(nwait, worker_flops, nworkers; nsamples=1000)
    rv = 0.0
    for _ in 1:nsamples
        s = NonIDOrderStatistic([gamma_worker_distribution(worker_flops) for _ in 1:nworkers], nwait)
        rv += rand(s)
    end
    rv / nsamples
end

### non-iid shifted exponential model

"""

Fit a `ShiftedExponential` distribution to the latency of each worker for each job.
"""
function shiftexp_df(df)
    df = filter([:nwait, :nworkers] => (x,y)->x==y, df)
    rv = DataFrame()
    row = Dict{String, Any}()
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]    
    for jobid in unique(df.jobid)
        dfi = filter(:jobid => (x)->x==jobid, df)
        nworkers = dfi.nworkers[1]
        row["nworkers"] = nworkers
        row["worker_flops"] = dfi.worker_flops[1]
        row["nbytes"] = dfi.nbytes[1]
        for i in 1:nworkers
            se = Distributions.fit(ShiftedExponential, float.(dfi[:, latency_columns[i]]))
            row["shift"], row["scale"] = params(se)
            row["worker_index"] = i
            push!(rv, row, cols=:union)
        end
    end
    rv
end

"""

Fit a two-dimensional `MvNormal` distribution to the shift and scale of the worker latency 
distribution for each unique value of `worker_flops`.
"""
function meta_shiftexp_df(dfse)
    rv = DataFrame()
    row = Dict{String, Any}()    
    for worker_flops in unique(dfse.worker_flops)
        dfi = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), dfse)
        d = Distributions.fit(MvNormal, hcat(dfi.shift, dfi.scale)')
        row["worker_flops"] = worker_flops
        row["nbytes"] = dfi.nbytes[1]
        row["mean_shift"] = d.μ[1]
        row["mean_scale"] = d.μ[2]
        row["var_shift"] = d.Σ[1, 1]
        row["var_scale"] = d.Σ[2, 2]
        row["cov"] = d.Σ[1, 2]
        push!(rv, row, cols=:union)
    end
    rv
end

function noise_df(dfse; nbins=100, minsamples=100)
    rv = DataFrame()
    row = Dict{String,Any}()
    # c = mean(dfse.scale.^2 ./ (dfse.shift .+ dfse.scale))
    edges = 10.0.^range(log10(minimum(dfse.shift)), log10(maximum(dfse.shift)), length=nbins+1)
    for i in 1:nbins
        dfi = filter(:shift => (x)->x>=edges[i], dfse)
        dfi = filter(:shift => (x)->x<edges[i+1], dfi)
        if size(dfi, 1) < minsamples
            continue
        end
        println(size(dfi, 1))
        row["nsamples"] = size(dfi, 1)
        row["shift"] = mean(dfi.shift)
        row["scale"] = mean(dfi.scale)
        noise = dfi.scale.^2 ./ (dfi.shift .+ dfi.scale)
        d = Distributions.fit(ShiftedExponential, noise)
        row["noise_shift"], row["noise_scale"] = params(d)
        push!(rv, row, cols=:union)
    end
    rv
end

function plot_noise(dfse; nbins=50)
    dfn = noise_df(dfse; nbins)
    plt.figure()
    plt.subplot(1, 2, 1)
    xs = dfn.shift .+ dfn.scale
    plt.plot(xs, -dfn.noise_shift, ".")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(xs, dfn.noise_scale, ".")    
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()    

    plt.tight_layout()
    return
end

function plot_meta_shiftexp(dfn)
    plt.figure()
    plt.subplot(3, 2, 1)
    xs = dfn.worker_flops
    ys = dfn.mean_shift
    plt.plot(xs, ys, ".")
    plt.ylabel("μ(shift)")
    plt.xlabel("nflops")
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")

    p, _ = fit_polynomial(xs, ys, 1)
    plt.plot(xs, p.(xs), "k-")

    plt.subplot(3, 2, 2)
    xs = dfn.worker_flops
    # xs = dfn.var_shift
    ys = sqrt.(dfn.mean_scale)
    plt.plot(xs, ys, ".")
    plt.ylabel("μ(scale)")
    plt.xlabel("nflops")    
    plt.grid()
    # plt.xscale("log")
    # plt.yscale("log")  

    p, _ = fit_polynomial(xs, ys, 1)
    plt.plot(xs, p.(xs), "k-")    

    plt.subplot(3, 2, 3)
    xs = dfn.worker_flops
    ys = sqrt.(dfn.var_shift)
    plt.plot(xs, ys, ".")
    plt.ylabel("σ(shift)")
    plt.xlabel("nflops")            
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")

    p, _ = fit_polynomial(xs, ys, 1)
    plt.plot(xs, p.(xs), "k-")    

    plt.subplot(3, 2, 4)
    xs = dfn.worker_flops
    ys = sqrt.(dfn.var_scale)
    plt.plot(xs, ys, ".")
    plt.ylabel("σ(scale)")
    plt.xlabel("nflops")        
    plt.grid()
    # plt.xscale("log")
    # plt.yscale("log")

    p, _ = fit_polynomial(xs, ys, 1)
    plt.plot(xs, p.(xs), "k-")    

    plt.subplot(3, 2, 5)
    plt.plot(dfn.worker_flops, dfn.cov, ".")
    plt.ylabel("cov(scale, shift)")
    plt.xlabel("nflops")        
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")

    plt.subplot(3, 2, 6)
    ys = dfn.cov ./ (sqrt.(dfn.var_shift) .* sqrt.(dfn.var_scale))
    plt.plot(dfn.worker_flops, ys, ".")
    plt.ylabel("cor(scale, shift)")
    plt.xlabel("nflops")        
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")    

    plt.tight_layout()
    return
end

# function non_iid_shifted_exponential_fit(df; worker_flops=1.14e7)
#     df = filter([:nwait, :nworkers] => (x,y)->x==y, df)
#     df = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), df)
#     rv = Vector{ShiftedExponential}(undef, 0)
#     for jobid in unique(df.jobid)
#         dfi = filter(:jobid => (x)->x==jobid, df)
#         se = Distributions.fit(ShiftedExponential, dfi.latency_worker_2)
#         push!(rv, se)
#     end
#     rv
# end

function non_iid_shifted_exponential_plot(rvs)
    ps = params.(rvs)
    shifts = [p[1] for p in ps]
    scales = [p[2] for p in ps]
    
    # plt.figure()
    # plt.plot(shifts, scales, ".")
    # return

    samples = zeros(2, length(rvs))
    samples[1, :] .= shifts
    samples[2, :] .= scales
    return Distributions.fit(MvNormal, samples)
    
    # shifts    
    plt.figure()
    xs = [p[1] for p in ps]
    sort!(xs)
    ys = range(0, 1, length=length(ps))   
    plt.plot(xs, ys)

    rv = Distributions.fit(Gamma, xs)
    println(rv)
    plt.plot(xs, cdf.(rv, xs), "k--")

    plt.title("Shift")

    # scales
    plt.figure()
    xs = [p[2] for p in ps]
    sort!(xs)
    ys = range(0, 1, length=length(ps))
    plt.plot(xs, ys)

    rv = Distributions.fit(Gamma, xs)
    println(rv)
    plt.plot(xs, cdf.(rv, xs), "k--")    

    plt.title("Scale")        
end

