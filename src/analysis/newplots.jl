### latency timeseries

"""

Plot the iteration latency of workers with indices in `workers` of job `jobid`.
"""
function plot_timeseries(df; jobid=rand(unique(df.jobid)), workers=[1, 11], separate=true)
    println("jobid: $jobid")
    df = filter(:jobid => (x)->x==jobid, df)
    plt.figure()
    for worker in workers
        xs = df.iteration        
        if separate
            # compute
            ys = df[:, "compute_latency_worker_$worker"]
            plt.plot(xs, ys, label="Worker $worker (comp.)")
            write_table(xs[1:100], ys[1001:1100], "timeseries_compute_$(jobid)_$(worker).csv")

            # communication
            ys = df[:, "latency_worker_$worker"] .- df[:, "compute_latency_worker_$worker"]
            plt.plot(xs, ys, label="Worker $worker (comm.)")
            write_table(xs[1:100], ys[1001:1100], "timeseries_communication_$(jobid)_$(worker).csv")
        end
        ys = df[:, "latency_worker_$worker"]
        plt.plot(xs, ys, label="Worker $worker")
        write_table(xs[1:100], ys[1001:1100], "timeseries_$(jobid)_$(worker).csv", nsamples=600)
        
    end    
    plt.grid()
    plt.legend()
    plt.title("Job $jobid")
    plt.xlabel("Iteration")
    plt.ylabel("Per-worker iteration latency [s]")
    plt.tight_layout()
    return
end

"""

Plot the cumulative time for a particular job.
"""
function plot_cumulative_time(df; jobid=rand(unique(df.jobid)), dfg=nothing, maxiterations=100)
    
    # empiric latency
    df = filter(:jobid => (x)->x==jobid, df)
    sort!(df, :iteration)
    if !isinf(maxiterations)
        df = filter(:iteration => (x)->x<=(maxiterations+1), df)
    end

    nwait = df.nwait[1]
    nworkers = df.nworkers[1]
    nflops = df.worker_flops[1]
    println("jobid: $jobid, nworkers: $nworkers, nwait: $nwait, nflops: $(round(nflops, sigdigits=3))")

    plt.figure()
    xs = df.iteration[1:end-1]
    ys = cumsum(df.latency[2:end] .+ df.update_latency[2:end])
    plt.plot(xs, ys, ".", label="Empiric")
    write_table(xs, ys, "cumulative_time_$(jobid).csv")

    # predicted latency
    if !isnothing(dfg)
        update_latency = mean(df.update_latency)
        ds_comm, ds_comp, ds_total = fit_worker_distributions(dfg; jobid)
        if any(isnothing.(ds_comm) .& isnothing.(ds_comp))
            println("Simulating iteration latency based on distributions fitted to the total latency")

            # not accounting for interaction between iterations
            spl = NonIDOrderStatistic(ds_total, nwait)
            xs = 1:size(df, 1)
            ys = cumsum(rand(spl, length(xs)) .+ update_latency)
            plt.plot(xs, ys, "k--", label="Int. not accounted for")
            write_table(xs, ys, "cumulative_time_sim_$(jobid).csv")

            # accounting for the interaction
            dfs = simulate_iterations(;nwait, ds_comm, ds_comp=ds_total, update_latency, niterations=100)
            xs = dfs.iteration
            ys = dfs.time
            plt.plot(xs, ys, "m--", label="Int. accounted for")
            write_table(xs, ys, "cumulative_time_simint_$(jobid).csv")
        else
            println("Simulating iteration latency based on separate distributions fitted to the communication and compute latency")

            # not accounting for interaction between iterations
            spl_comm = NonIDOrderStatistic(ds_comm, nwait)
            spl_comp = NonIDOrderStatistic(ds_comp, nwait)
            xs = 1:size(df, 1)
            ys = cumsum(rand(spl_comm, length(xs)) .+ rand(spl_comp, length(xs)) .+ update_latency)
            plt.plot(xs, ys, "k--", label="Int. not accounted for")

            # accounting for the interaction
            dfs = simulate_iterations(;nwait, ds_comm, ds_comp, update_latency)
            xs = dfs.iteration
            ys = dfs.time
            plt.plot(xs, ys, "m--", label="Int. accounted for")
        end
    end

    plt.grid()
    plt.legend()
    plt.title("Job $jobid")
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative time [s]")
    plt.tight_layout()
    return    
end

### order statistics

"""

Plot order statistics latency for a given computational load.
"""
function plot_orderstats(df; nworkers=nothing, worker_flops=nothing, nbytes=30048, deg3m=nothing, osm=nothing, niidm=nothing)
    if !isnothing(nworkers)
        df = filter(:nworkers => (x)->x==nworkers, df)
    end
    if !isnothing(worker_flops)
        df = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), df)
    end
    df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    df = filter(:iteration => (x)->x>1, df)
    df = filter(:nbytes => (x)->x==nbytes, df)
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
        for worker_flops in sort!(unique(dfi.worker_flops))
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
            plt.plot(xs, ys, "-o", label="$nworkers workers, $(round(worker_flops, sigdigits=3)) workload")
            write_table(xs, ys, "orderstats_$(nworkers)_$(worker_flops).csv")

            println("Acuteness: $(ys[end] / ys[1])")

            # # latency predicted by the degree-3 model (local)
            # p, _ = fit_polynomial(xs, ys, 3)            
            # ys = p.(1:nworkers)
            # plt.plot(xs, ys, "c--")
            # write_table(xs, ys, "./results/orderstats_deg3l_$(nworkers)_$(worker_flops).csv")

            # latency predicted by the degree-3 model (global)
            if !isnothing(deg3m)
                ys = predict_latency.(1:nworkers, worker_flops, nworkers; deg3m)
                plt.plot(xs, ys, "k--")
                write_table(xs, ys, "orderstats_deg3_$(nworkers)_$(worker_flops).csv")
            end

            # # latency predicted by the shifted exponential model
            # ys = predict_latency_shiftexp.(1:nworkers, worker_flops, nworkers)
            # plt.plot(xs, ys, "m--")
            # write_table(xs, ys, "./results/orderstats_shiftexp_$(nworkers)_$(worker_flops).csv")

            # latency predicted by the non-iid order statistics model
            if !isnothing(osm)
                ys = predict_latency_gamma(worker_flops, nworkers; osm)
                plt.plot(xs, ys, "m-")
            end

            # latency predicted by the new non-iid model
            if !isnothing(niidm)
                ys = predict_latency_niid(nbytes, worker_flops, nworkers; niidm)
                plt.plot(xs, ys, "c--")
            end
        end
    end
    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.xlabel("Order")        
    plt.ylabel("Latency [s]")
    # plt.tight_layout()
    return
end

"""

Plot order statistics latency for a particular job.
"""
function plot_orderstats(df, jobid)    
    df = filter(:jobid => (x)->x==jobid, df)
    nbytes, nflops = df.nbytes[1], df.worker_flops[1]
    println("nbytes: $nbytes, nflops: $nflops")
    nworkers = df.nworkers[1]
    niterations = df.niterations[1]
    orderstats = zeros(nworkers)
    buffer = zeros(nworkers)
    latency_columns = ["latency_worker_$(i)" for i in 1:nworkers]    
    compute_latency_columns = ["compute_latency_worker_$(i)" for i in 1:nworkers]    

    plt.figure()    

    # empirical orderstats
    for i in 1:niterations
        for j in 1:nworkers
            buffer[j] = df[i, latency_columns[j]]
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, label="Empirical")
    write_table(1:nworkers, orderstats, "orderstats_$(jobid).csv")

    # # upper bound based on considering only the w fastest workers
    # ds = [Distributions.fit(ShiftedExponential, df[:, latency_columns[j]]) for j in 1:nworkers]
    # sort!(ds, by=mean) # sort by mean total latency
    # f = (x, w) -> reduce(*, [cdf(ds[i], x) for i in 1:w])
    # ts = range(quantile(ds[1], 0.1), quantile(ds[end], 0.99999), length=100)
    # orderstats = [ts[searchsortedfirst(f.(ts, w), 0.5)] for w in 1:nworkers]
    # plt.plot(1:nworkers, orderstats, ".", label="Bound")

    # global mean and variance
    m_comm, v_comm = 0.0, 0.0
    m_comp, v_comp = 0.0, 0.0
    for j in 1:nworkers
        vs = df[:, latency_columns[j]] .- df[:, compute_latency_columns[j]]
        m_comm += mean(vs)
        v_comm += var(vs)
        vs = df[:, compute_latency_columns[j]]
        m_comp += mean(vs)
        v_comp += var(vs)
    end
    m_comm /= nworkers
    v_comm /= nworkers
    m_comp /= nworkers
    v_comp /= nworkers

    # global gamma
    θ = v_comm / m_comm
    α = m_comm / θ
    d_comm = Gamma(α, θ)
    θ = v_comp / m_comp
    α = m_comp / θ
    d_comp = Gamma(α, θ)    
    orderstats .= 0
    for i in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(d_comm) + rand(d_comp)
        end
        sort!(buffer)
        orderstats += buffer
    end    
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, "--", label="Global Gamma")
    write_table(1:nworkers, orderstats, "orderstats_global_gamma_$(jobid).csv")

    # global shiftexp
    θ = sqrt(v_comm)
    s = m_comm - θ
    d_comm = ShiftedExponential(s, θ)
    θ = sqrt(v_comp)
    s = m_comp - θ
    d_comp = ShiftedExponential(s, θ)
    orderstats .= 0
    for i in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(d_comm) + rand(d_comp)
        end        
        sort!(buffer)
        orderstats += buffer
    end    
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, "--", label="Global ShiftExp")
    write_table(1:nworkers, orderstats, "orderstats_global_shiftexp_$(jobid).csv")

    # independent orderstats (gamma)
    ds = [Distributions.fit(Gamma, df[:, latency_columns[j]]) for j in 1:nworkers]
    orderstats .= 0
    for _ in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(ds[j])
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations    
    plt.plot(1:nworkers, orderstats, "k--", label="Gamma")

    # independent orderstats (shiftexp)
    ds = [Distributions.fit(ShiftedExponential, df[:, latency_columns[j]]) for j in 1:nworkers]
    orderstats .= 0
    for _ in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(ds[j])
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations    
    plt.plot(1:nworkers, orderstats, "r--", label="ShiftExp")    

    # independent orderstats w. separate communication and compute
    ds_comm = [Distributions.fit(ShiftedExponential, df[:, latency_columns[j]] .- df[:, compute_latency_columns[j]]) for j in 1:nworkers]
    ds_comp = [Distributions.fit(ShiftedExponential, df[:, compute_latency_columns[j]]) for j in 1:nworkers]
    orderstats .= 0
    for _ in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(ds_comm[j]) + rand(ds_comp[j])
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, "b--", label="ShiftExp-ShiftExp (ind., sep.)")    
    write_table(1:nworkers, orderstats, "orderstats_shiftexp_shiftexp_$(jobid).csv")        

    # independent orderstats w. separate communication and compute
    ds_comm = [Distributions.fit(ShiftedExponential, df[:, latency_columns[j]] .- df[:, compute_latency_columns[j]]) for j in 1:nworkers]
    ds_comp = [Distributions.fit(Gamma, df[:, compute_latency_columns[j]]) for j in 1:nworkers]
    orderstats .= 0
    for _ in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(ds_comm[j]) + rand(ds_comp[j])
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, "--", label="ShiftExp-Gamma")
    
    # independent orderstats w. separate communication and compute
    ds_comm = [Distributions.fit(Gamma, df[:, latency_columns[j]] .- df[:, compute_latency_columns[j]]) for j in 1:nworkers]
    ds_comp = [Distributions.fit(ShiftedExponential, df[:, compute_latency_columns[j]]) for j in 1:nworkers]
    orderstats .= 0
    for _ in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(ds_comm[j]) + rand(ds_comp[j])
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, "--", label="Gamma-ShiftExp")      

    # independent orderstats w. separate communication and compute
    ds_comm = [Distributions.fit(Gamma, df[:, latency_columns[j]] .- df[:, compute_latency_columns[j]]) for j in 1:nworkers]
    ds_comp = [Distributions.fit(Gamma, df[:, compute_latency_columns[j]]) for j in 1:nworkers]
    orderstats .= 0
    for _ in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(ds_comm[j]) + rand(ds_comp[j])
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, "c--", label="Gamma-Gamma (ind., sep.)")    
    write_table(1:nworkers, orderstats, "orderstats_gamma_gamma_$(jobid).csv")

    # dependent orderstats (using a Normal copula)
    μ = zeros(nworkers)
    Σ = Matrix(1.0.*I, nworkers, nworkers)
    for i in 1:nworkers
        vsi = df[:, latency_columns[i]] .- df[:, compute_latency_columns[i]]
        for j in (i+1):nworkers
            vsj = df[:, latency_columns[j]] .- df[:, compute_latency_columns[j]]
            # Σ[i, j] = 0.4
            Σ[i, j] = cor(vsi, vsj)
        end
    end

    ## fix non-positive-definite matrices
    if !isposdef(Symmetric(Σ))
        F = eigen(Symmetric(Σ))
        replace!((x)->max(sqrt(eps(Float64)), x), F.values)
        Σ = F.vectors*Diagonal(F.values)*F.vectors'
    end

    # copula
    ds_comm = [Distributions.fit(Gamma, df[:, latency_columns[j]] .- df[:, compute_latency_columns[j]]) for j in 1:nworkers]
    ds_comp = [Distributions.fit(Gamma, df[:, compute_latency_columns[j]]) for j in 1:nworkers]    
    copula = MvNormal(μ, Symmetric(Σ))
    normal = Normal() # standard normal
    sample = zeros(nworkers)
    orderstats .= 0
    for _ in 1:niterations
        Distributions.rand!(copula, sample) # sample from the MvNormal
        for j in 1:nworkers
            buffer[j] = quantile(ds_comm[j], cdf(normal, sample[j])) # Convert to uniform and then to the correct marginal
            buffer[j] += rand(ds_comp[j]) # add compute latency
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, "m--", label="Simulated (copula)")
    write_table(1:nworkers, orderstats, "orderstats_gamma_gamma_copula_$(jobid).csv")

    plt.legend()
    plt.grid()
    plt.xlabel("Order")
    plt.ylabel("Latency [s]")
    return
end

"""

Plot the CDF of the `w`-th order statistic in the `iter`-th iteration.
"""
function plot_orderstats_distribution(df, w; nbytes=30048, nflops, iterspacing=10, nworkers=72)
    df = filter(:nbytes => (x)->x==nbytes, df)
    df = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), df)
    df = filter(:nworkers => (x)->x==nworkers, df)
    df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    if size(df, 1) == 0
        error("no rows match nbytes: $nbytes and nflops: $nflops")
    end
    latency_columns = ["latency_worker_$(i)" for i in 1:nworkers]
    buffer = zeros(nworkers)
    xs = zeros(0)
    jobids = unique(df.jobid)
    println("Computing CDF over $(length(jobids)) jobs")
    for jobid in jobids
        dfi = filter(:jobid=>(x)->x==jobid, df)
        sort!(dfi, :iteration)
        for i in iterspacing:iterspacing:dfi.niterations[1]
            for j in 1:nworkers
                buffer[j] = dfi[i, latency_columns[j]]
            end
            sort!(buffer)
            push!(xs, buffer[w])            
        end
    end
    sort!(xs)
    ys = range(0, 1, length=length(xs))
    plt.figure()
    plt.plot(xs, ys)
    plt.grid()
    return
end

### prior distribution

"""

Plot the empirical CDF of the `iter`-th iteration computed over worker realizations.
"""
function plot_prior_latency_distribution(df; nbytes=30048, nflops, iter=10)
    df = filter(:nbytes => (x)->x==nbytes, df)
    df = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), df)
    if size(df, 1) == 0
        error("no rows match nbytes: $nbytes and nflops: $nflops")
    end
    latency_columns = ["latency_worker_$(i)" for i in 1:maximum(df.nworkers)]
    # buffer = zeros(nworkers)
    xs = zeros(0)
    jobids = unique(df.jobid)
    println("Computing CDF over $(length(jobids)) jobs")
    for jobid in jobids
        dfi = filter(:jobid=>(x)->x==jobid, df)
        nworkers = dfi.nworkers[1]
        sort!(dfi, :iteration)
        for j in 1:nworkers
            push!(xs, dfi[iter, latency_columns[j]])
        end
    end    
    sort!(xs)
    ys = range(0, 1, length=length(xs))
    plt.figure()
    plt.plot(xs, ys)
    plt.grid()
    return    
end

"""

Plot the average orderstats of the `iter`-th iteration computed over worker realizations.

iters=10:10:100
"""
function plot_prior_orderstats(df; nworkers, nwait=nworkers, nbytes=30048, nflops, iters=10, niidm=nothing)
    df = filter(:nworkers => (x)->x==nworkers, df)
    df = filter(:nwait => (x)->x==nwait, df)
    df = filter(:nbytes => (x)->x==nbytes, df)    
    df = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), df)
    if size(df, 1) == 0
        error("no rows match nbytes: $nbytes and nflops: $nflops")
    end
    latency_columns = ["latency_worker_$(i)" for i in 1:maximum(df.nworkers)]
    repoch_columns = ["repoch_worker_$(i)" for i in 1:maximum(df.nworkers)]

    # plot empirical prior orderstats when all workers are available
    buffer = zeros(nworkers)
    orderstats = zeros(nwait)
    jobids = unique(df.jobid)
    nsamples = 0
    println("Computing orderstats over $(length(jobids)) jobs")
    for jobid in jobids
        dfi = filter(:jobid=>(x)->x==jobid, df)
        sort!(dfi, :iteration)
        for i in iters
            if i > size(dfi, 1)
                continue
            end
            for j in 1:nworkers
                if dfi[i, repoch_columns[j]] == dfi[i, :iteration]
                    buffer[j] = dfi[i, latency_columns[j]]
                else
                    buffer[j] = Inf
                end
            end
            sort!(buffer)
            orderstats += view(buffer, 1:nwait)
            nsamples += 1
        end
    end    
    orderstats ./= nsamples
    xs = 1:nwait
    plt.figure()
    plt.plot(xs, orderstats, "-o")
    write_table(xs, orderstats, "prior_orderstats_$(nworkers)_$(nbytes)_$(round(nflops, sigdigits=3)).csv")

    # latency predicted by the new non-iid model
    if !isnothing(niidm)
        xs = 1:nworkers
        ys = predict_latency_niid(nbytes, nflops, nworkers; niidm)
        plt.plot(xs, ys, "c--")
        write_table(xs, ys, "prior_orderstats_niidm_$(nworkers)_$(nbytes)_$(round(nflops, sigdigits=3)).csv")
    end

    plt.grid()
    return    
end

### worker-worker latency correlation

"""

Plot the CDF of the correlation between pairs of workers.
"""
function plot_worker_latency_cov_cdf(df; nflops, nbytes=30048, maxworkers=108, latency="total", minsamples=10)
    df = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), df)
    df = filter(:nbytes => (x)->x==nbytes, df)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:maxworkers]
    xs = zeros(0)
    xsr = zeros(0)
    for jobid in unique(df.jobid)
        dfi = filter(:jobid => (x)->x==jobid, df)
        if size(dfi, 1) < minsamples
            continue
        end
        nworkers = min(maxworkers, dfi.nworkers[1])
        for i in 1:nworkers
            if latency == "total"
                xsi = float.(dfi[:, latency_columns[i]])
            elseif latency == "communication"
                xsi = dfi[:, latency_columns[i]] .- dfi[:, compute_latency_columns[i]]            
            elseif latency == "compute"
                xsi = dfi[:, compute_latency_columns[i]]
            else
                error("latency must be in [total, communication, compute]")
            end
            if minimum(xsi) <= 0
                continue
            end
            try
                rvi = Distributions.fit(Gamma, xsi)
            catch e
                return xsi
            end
            rvi = Distributions.fit(Gamma, xsi)
            rsamplesi = rand(rvi, size(dfi, 1))
            for j in 1:nworkers
                if j == i
                    continue
                end
                if latency == "total"
                    xsj = float.(dfi[:, latency_columns[j]])
                elseif latency == "communication"
                    xsj = dfi[:, latency_columns[j]] .- dfi[:, compute_latency_columns[j]]            
                elseif latency == "compute"
                    xsj = dfi[:, compute_latency_columns[j]]
                else
                    error("latency must be in [total, communication, compute]")
                end
                if minimum(xsj) <= 0
                    continue
                end
                push!(xs, cor(xsi, xsj))
                
                rvj = Distributions.fit(Gamma, xsj)
                rsamplesj = rand(rvj, size(dfi, 1))
                push!(xsr, cor(rsamplesi, rsamplesj))
            end
        end
    end

    # empirical cdf
    plt.figure()    
    sort!(xs)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys)
    write_table(xs, ys, "cov_cdf_$(latency)_$(round(nflops, sigdigits=3))_$(nbytes).csv")

    # independent cdf
    sort!(xsr)
    plt.plot(xsr, ys, "k--")
    write_table(xsr, ys, "cov_cdf_ind_$(latency)_$(round(nflops, sigdigits=3))_$(nbytes).csv")
    return
end

function plot_worker_latency_cov_old(df; jobid=rand(unique(df.jobid)), worker_indices=1:5)
    df = filter(:jobid => (x)->x==jobid, df)
    worker_flops = df.worker_flops[1]
    nbytes = df.nbytes[1]
    nworkers = length(worker_indices)
    latency_columns = ["compute_latency_worker_$i" for i in worker_indices]
    
    plt.figure()
    plt.title("job $jobid ($(round(worker_flops, sigdigits=3)) flops, $nbytes bytes, sparse matrix)")    
    for i in 1:nworkers
        for j in 1:nworkers
            plt.subplot(nworkers, nworkers, (i-1)*nworkers+j)
            plt.plot(df[:, latency_columns[i]], df[:, latency_columns[j]], ".")            
            plt.axis("equal")
        end
    end
    plt.tight_layout()
    return
end

### auto-correlation

"""

Plot the latency auto-correlation function averaged over many realizations of the process.
"""
function plot_autocorrelation(df; nflops, nbytes=30048, maxlag=100, latency="total")
    df = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), df)
    df = filter(:nbytes => (x)->x==nbytes, df)
    sort!(df, [:jobid, :iteration])    
    maxworkers = maximum(df.nworkers)
    ys = zeros(maxlag)
    nsamples = zeros(Int, maxlag)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:maxworkers]
    for jobid in unique(df.jobid)
        dfi = filter(:jobid => (x)->x==jobid, df)
        nworkers = dfi.nworkers[1]
        lags = 0:min(maxlag-1, size(dfi, 1)-1)
        for i in 1:nworkers
            if latency == "total"
                vs = float.(dfi[:, latency_columns[i]]) # total latency
            elseif latency == "communication"
                vs = float.(dfi[:, latency_columns[i]] .- dfi[:, compute_latency_columns[i]]) # communication latency
            elseif latency == "compute"
                vs = float.(dfi[:, compute_latency_columns[i]]) # compute latency
            else
                error("latency must be one of [total, communication, compute]")
            end
            ys[lags.+1] .+= autocor(vs, lags)
            nsamples[lags.+1] .+= 1
        end
    end
    ys ./= nsamples
    plt.figure()
    xs = 0:(maxlag-1)
    plt.plot(xs, ys)
    write_table(xs, ys, "ac_$(latency)_$(round(nflops, sigdigits=3))_$(nbytes).csv", nsamples=maxlag)
    plt.xlabel("Lag (iterations)")
    plt.ylabel("Auto-correlation")
    return
end

# I want to measure to what extent the latency of subsequent iterations are independent
# And I want to average this over many runs

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
function predict_latency(nwait, nflops, nworkers; deg3m)
    b1, c1, d1, e1, b2, c2, d2, e2 = deg3m
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

function plot_deg3_model(df3, deg3m=nothing)

    # coefficients of global model
    if !isnothing(deg3m)
        nflops1 = 10 .^ range(log10(minimum(df3.worker_flops)), log10(maximum(df3.worker_flops)), length=100)
        nflops2 = 10 .^ range(log10(minimum(df3.worker_flops./df3.nworkers)), log10(maximum(df3.worker_flops./df3.nworkers)), length=100)
        nflops3 = 10 .^ range(log10(minimum(df3.worker_flops./df3.nworkers.^2)), log10(maximum(df3.worker_flops./df3.nworkers.^2)), length=100)
        nflops4 = 10 .^ range(log10(minimum(df3.worker_flops./df3.nworkers.^3)), log10(maximum(df3.worker_flops./df3.nworkers.^3)), length=100)
        # b1, c1, d1, e1, b2, c2, d2, e2 = deg3_coeffs("c5xlarge")
        b1, c1, d1, e1, b2, c2, d2, e2 = deg3m
        α1 = b1 .+ b2.*nflops1
        α2 = c1 .+ c2.*nflops2
        α3 = -(d1 .+ d2.*nflops3)
        α4 = e1 .+ e2.*nflops4
        coefficients = [α1, α2, α3, α4]
        nflops_all = [nflops1, nflops2, nflops3, nflops4]
    end

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
            plt.plot(xs, ys, ".", label="Local model ($nworkers workers)")
            write_table(xs, ys, "./results/deg3_$(col)_$(nworkers).csv")
        end

        # coefficients of the global model
        if !isnothing(deg3m)
            xs = nflops_all[i]
            ys = coefficients[i]
            plt.plot(xs, ys, "k-", label="Global model")
            write_table(xs, ys, "./results/deg3_global_$(col).csv")
        end

        if i == 3
            plt.ylabel("-$col")
        else
            plt.ylabel(col)
        end
        if i == 1
            plt.legend()
        end
        plt.xlabel("c / nworkers^$(i-1)")
        plt.grid()
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
function plot_predictions(c0=1.6362946777247114e9; df=nothing, maxworkers=200, deg3m)
    if !isnothing(df)
        df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    end
    nworkers = 1:maxworkers
    c = c0 ./ nworkers
    plt.figure()
    for ϕ in [0.5, 1.0]
        nwait = ϕ.*nworkers
        xs = nworkers
        ys = predict_latency.(nwait, c, nworkers; deg3m)
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

### convergence plots

"""

Plot the rate of convergence over time for DSAG, SAG, SGD, and coded computing. Let 
`latency=empirical` to plot against empirical latency, or let `latency=c5xlarge` to plot against 
latency computed by the model, fitted to traces recorded on `c5xlarge` instances.
"""
function plot_convergence(df, nworkers, opt=maximum(skipmissing(df.mse)); latency="empirical", niidm=nothing)
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

    upscale = 2
    nbytes = 30048

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
        # for simulations
        nbytes = dfj.nbytes[1]
        nflops = dfj.worker_flops[1]
        # return nbytes, nflops
        # update_latency = mean(dfj.update_latency)

        println("DSAG: $(length(unique(dfj.jobid))) jobs")
        if size(dfj, 1) > 0
            dfj = combine(groupby(dfj, :iteration), :mse => mean => :mse, :time => mean => :time)
            if latency == "empirical"
                println("Plotting DSAG with empirical latency")
            else
                # dfj.time .= predict_latency(nwait, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration
                dfc_comm, dfc_comp = niidm
                dfs = simulate_iterations(nbytes, nflops/upscale; niterations=maximum(dfj.iteration), nworkers=nworkers*upscale, nwait=nwait*upscale, dfc_comm, dfc_comp, update_latency=0.0022031946363636366)
                ys = opt .- dfj.mse
                dfj.time .= dfs.time[dfj.iteration]
                println("Plotting DSAG with model latency for $latency")
            end
            xs = dfj.time
            ys = opt.-dfj.mse
            plt.semilogy(xs, ys, ".-", label="DSAG w=$nwait, p=$nsubpartitions")
            filename = "dsag_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"            
            write_table(xs, ys, filename)
        end
        println()

        # simulated latency (using the event-driven model)
        if nwait == 1

            # # latency predicted by the new non-iid model
            # t_iter = predict_latency_niid(nbytes, nflops, nworkers; niidm)[nwait]
            # # t_iter = predict_latency(Nw, worker_flops, nworkers)
            # xs = (t_iter + 0.0022031946363636366) .* dfj.iteration
            # ys = opt .- dfj.mse
            # plt.plot(xs, ys, "k.")

            dfc_comm, dfc_comp = niidm
            dfs = simulate_iterations(nbytes, nflops; niterations=maximum(dfj.iteration), nworkers, nwait, dfc_comm, dfc_comp, update_latency=0.0022031946363636366)
            ys = opt .- dfj.mse
            xs = dfs.time[dfj.iteration]
            plt.plot(xs, ys, "k.")
            filename = "event_driven_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"
            write_table(xs, ys, filename)
        end

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
        # dfj.time .= predict_latency(nworkers, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration

        dfc_comm, dfc_comp = niidm
        nflops = mean(dfi.worker_flops)        
        dfs = simulate_iterations(nbytes, nflops/upscale; balanced=true, nruns=50, niterations=maximum(dfj.iteration), nworkers=nworkers*upscale, nwait=nworkers*upscale, dfc_comm, dfc_comp, update_latency=0.0022031946363636366)
        ys = opt .- dfj.mse
        dfj.time .= dfs.time[dfj.iteration]

        println("Plotting SAG with model latency for $latency")
    end
    if size(dfj, 1) > 0
        xs = dfj.time
        ys = opt.-dfj.mse
        plt.semilogy(xs, ys, "o-", label="SAG p=$nsubpartitions")
        filename = "sag_$(nworkers)_$(nsubpartitions)_$(stepsize).csv"
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
        # dfj.time .= predict_latency(nworkers, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration

        dfc_comm, dfc_comp = niidm
        nflops = mean(dfi.worker_flops)        
        dfs = simulate_iterations(nbytes, nflops/upscale; niterations=maximum(dfj.iteration), nworkers=nworkers*upscale, nwait=nworkers*upscale, dfc_comm, dfc_comp, update_latency=0.0022031946363636366)
        ys = opt .- dfj.mse
        dfj.time .= dfs.time[dfj.iteration]

        println("Plotting SGD with model latency for $latency")
    end    
    if size(dfj, 1) > 0
        xs = dfj.time
        ys = opt.-dfj.mse
        plt.semilogy(xs, ys, "c^-", label="SGD p=$nsubpartitions")
        filename = "sgd_$(nworkers)_$(nsubpartitions)_$(stepsize).csv"
        write_table(xs, ys, filename)        
    end

    # # Plot GD
    # stepsize = 1.0
    # dfi = df
    # dfi = dfi[dfi.nwait .== nworkers, :]
    # dfi = dfi[dfi.nsubpartitions .== 1, :]
    # dfi = dfi[dfi.variancereduced .== false, :]
    # dfi = dfi[dfi.stepsize .== stepsize, :]
    # println("GD $(length(unique(dfi.jobid))) jobs")
    # dfj = by(dfi, :iteration, :mse => mean => :mse, :time => mean => :time)
    # if latency == "empirical"
    #     println("Plotting GD with empirical latency")
    # else
    #     # dfj.time .= predict_latency(nworkers, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration

    #     dfc_comm, dfc_comp = niidm
    #     nflops = mean(dfi.worker_flops)        
    #     dfs = simulate_iterations(nbytes, nflops/upscale; niterations=maximum(dfj.iteration), nworkers=nworkers*upscale, nwait=nworkers*upscale, dfc_comm, dfc_comp, update_latency=0.0022031946363636366)
    #     ys = opt .- dfj.mse
    #     dfj.time .= dfs.time[dfj.iteration]        

    #     println("Plotting GD with model latency for $latency")
    # end    
    # if size(dfj, 1) > 0
    #     xs = dfj.time
    #     ys = opt.-dfj.mse
    #     plt.semilogy(xs, ys, "ms-", label="GD")
    #     filename = "gd_$(nworkers)_$(stepsize).csv"
    #     write_table(xs, ys, filename)
    # end

    # # plot coded computing bound
    # if !isnothing(niidm)
    #     r = 2 # replication factor
    #     Nw = 1 # number of workers to wait for
    #     samp = 1 # workload up-scaling

    #     # get the average error per iteration of GD
    #     dfi = df
    #     dfi = dfi[dfi.nsubpartitions .== 1, :]
    #     dfi = dfi[dfi.nwait .== nworkers, :]
    #     dfi = dfi[dfi.stepsize .== 1, :]
    #     dfi = dfi[dfi.variancereduced .== false, :]
    #     dfi = dfi[dfi.nostale .== false, :]
    #     dfj = combine(groupby(dfi, :iteration), :mse => mean => :mse)
    #     sort!(dfj, :iteration)
    #     ys = opt .- dfj.mse

    #     # compute the iteration time for a scheme with a factor r replication
    #     @assert length(unique(dfi.worker_flops)) == 1
    #     worker_flops = r*mean(dfi.worker_flops)
    #     nbytes, = unique(dfi.nbytes)

    #     # latency predicted by the new non-iid model
    #     t_iter = predict_latency_niid(nbytes, worker_flops, nworkers; niidm)[Nw]
    #     # t_iter = predict_latency(Nw, worker_flops, nworkers)
    #     xs = t_iter .* dfj.iteration

    #     # # latency predicted by the event-driven model
    #     # dfc_comm, dfc_comp = niidm
    #     # dfs = simulate_iterations(nbytes, worker_flops; niterations=maximum(dfj.iteration), nworkers, nwait=Nw, dfc_comm, dfc_comp, update_latency=0.0022031946363636366)        
    #     # xs = dfs.time[dfj.iteration]
    #     # return dfs

    #     # make the plot
    #     plt.semilogy(xs, ys, "--k", label="Bound r: $r, Nw: $Nw")
    #     filename = "bound_$(nworkers)_$(stepsize).csv"
    #     write_table(xs, ys, filename)    
    # end

    plt.xlim(1e-2, 1e2)
    plt.xscale("log")
    plt.grid()
    plt.legend()    
    plt.xlabel("Time [s]")
    plt.ylabel("Explained Variance Sub-optimality Gap")
    return
end

### non-iid model

"""

Plot the latency distribution of individual workers.
"""
function plot_worker_latency_distribution(df; jobid=rand(unique(df.jobid)), worker_indices=[1, 11])
    df = filter(:jobid => (x)->x==jobid, df)
    worker_flops = df.worker_flops[1]
    nbytes = df.nbytes[1]
    plt.figure()
    plt.title("job $jobid ($(round(worker_flops, sigdigits=3)) flops, $nbytes bytes, sparse matrix)")    

    # overall
    plt.subplot(3, 1, 1)
    for i in worker_indices        
        xs = sort(df[:, "latency_worker_$(i)"])
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, label="Worker $i")
        write_table(xs, ys, "cdf_$(jobid)_$(i).csv")
        j = length(xs) * 0.01
        d = Distributions.fit(Gamma, xs[ceil(Int, j):end-floor(Int, j)])
        xs = range(quantile(d, 0.000001), quantile(d, 0.999999), length=100)
        ys = cdf.(d, xs)
        write_table(xs, ys, "cdf_fit_$(jobid)_$(i).csv")
        if i == worker_indices[end]
            plt.plot(xs, ys, "k--", label="Fitted Gamma dist.")
        else
            plt.plot(xs, ys, "k--")
        end        
    end
    # plt.ylim(1e-2, 1)
    plt.xlabel("Overall per-worker iteration latency [s]")
    plt.ylabel("CDF")
    plt.legend()
    # plt.xscale("log")
    # plt.yscale("log")
    plt.grid()    

    return

    # communication    
    plt.subplot(3, 1, 2)
    for i in worker_indices        
        xs = df[:, "latency_worker_$(i)"] .- df[:, "compute_latency_worker_$(i)"]
        sort!(xs)
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, label="Worker $i")
        write_table(xs, ys, "cdf_communication_$(jobid)_$(i).csv")
        # plt.hist(xs, 200, density=true)
        # j = round(Int, 0*length(xs))
        j = round(Int, length(xs) * 0.01)
        d = Distributions.fit(Gamma, xs[1:end-j])
        xs = range(quantile(d, 0.000001), quantile(d, 0.999999), length=100)
        ys = cdf.(d, xs)
        write_table(xs, ys, "cdf_fit_compute_$(jobid)_$(i).csv")
        if i == worker_indices[end]
            plt.plot(xs, ys, "k--", label="Fitted Gamma dist.")
        else
            plt.plot(xs, ys, "k--")
        end
    end
    # plt.ylim(1e-2, 1)
    plt.xlabel("Per-worker comm. latency [s]")
    plt.ylabel("CDF")
    # plt.legend()
    # plt.xscale("log")
    # plt.yscale("log")
    plt.grid()        

    # computation
    plt.subplot(3, 1, 3)
    for i in worker_indices
        xs = sort(df[:, "compute_latency_worker_$(i)"])
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, label="Worker $i")
        write_table(xs, ys, "cdf_compute_$(jobid)_$(i).csv")
        j = round(Int, 0.01*length(xs))
        d = Distributions.fit(Gamma, xs[j:end-j])
        xs = range(quantile(d, 0.000001), quantile(d, 0.999999), length=100)
        ys = cdf.(d, xs)
        write_table(xs, ys, "cdf_fit_communication_$(jobid)_$(i).csv")
        if i == worker_indices[end]
            plt.plot(xs, ys, "k--", label="Fitted Gamma dist.")
        else
            plt.plot(xs, ys, "k--")
        end
    end    
    # plt.ylim(1e-2, 1)
    plt.xlabel("Per-worker comp. latency [s]")
    plt.ylabel("CDF")
    # plt.legend()
    # plt.xscale("log")
    # plt.yscale("log")
    plt.grid()         

    plt.tight_layout()
    # plt.savefig("per_worker_distribution.png", dpi=600)    
    return
end

"""

Compute the mean and variance of the per-iteration latency for each job and worker.
"""
function worker_distribution_df(df; minsamples=100, prune_comm=0.02, prune=0.01)
    # df = filter([:nwait, :nworkers] => (x,y)->x==y, df)
    df = copy(df)
    rv = DataFrame()
    row = Dict{String, Any}()
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]    
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:maxworkers]    
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

            if "compute_latency_worker_1" in names(dfi)

                # compute latency                
                ys = float.(dfi[:, compute_latency_columns[i]])
                j1, j2 = ceil(Int, length(ys) * prune), floor(Int, length(ys) * prune)
                ys = ys[j1:end-j2]
                row["comp_mean"] = mean(ys)
                row["comp_var"] = var(ys)

                # communication latency
                ys = float.(dfi[:, latency_columns[i]] .- dfi[:, compute_latency_columns[i]])
                j1, j2 = ceil(Int, length(ys) * prune), floor(Int, length(ys) * prune)
                ys = ys[j1:end-j2]
                # if prune_comm != zero(prune_comm)
                #     j = round(Int, prune_comm*length(ys))
                #     sort!(ys)
                #     ys = ys[1:end-j]
                # end                
                row["comm_mean"] = mean(ys)
                row["comm_var"] = var(ys)
            else
                row["comp_mean"] = missing
                row["comp_var"] = missing
                row["comm_mean"] = missing
                row["comm_var"] = missing
            end

            # overall latency
            ys = float.(dfi[:, latency_columns[i]])
            j1, j2 = ceil(Int, length(ys) * prune), floor(Int, length(ys) * prune)
            ys = ys[j1:end-j2]
            row["mean"] = mean(ys)
            row["var"] = var(ys)

            row["worker_index"] = i
            push!(rv, row, cols=:union)
        end
    end
    rv
end

"""

Return distributions fit to the communication, compute, and total latency of each worker.
"""
function fit_worker_distributions(dfg; jobid)
    dfg = filter(:jobid => (x)->x==jobid, dfg)
    size(dfg, 1) > 0 || error("job $jobid doesn't exist")
    sort!(dfg, :worker_index)
    nworkers = dfg.nworkers[1]
    if ismissing(dfg.comp_mean[1])
        ds_comm = [nothing for _ in 1:nworkers]
        ds_comp = [nothing for _ in 1:nworkers]
    else
        # # ShiftedExponential model
        # θs = sqrt.(dfg.comm_var)
        # ss = dfg.comm_mean .- θs
        # ds_comm = ShiftedExponential.(ss, θs)
        # θs = sqrt.(dfg.comp_var)
        # ss = dfg.comp_mean .- θs
        # ds_comp = ShiftedExponential.(ss, θs)        

        # Gamma model
        θs = dfg.comm_var ./ dfg.comm_mean
        αs = dfg.comm_mean ./ θs
        ds_comm = Gamma.(αs, θs)            
        θs = dfg.comp_var ./ dfg.comp_mean
        αs = dfg.comp_mean ./ θs
        ds_comp = Gamma.(αs, θs)
    end
    # # ShiftedExponential model
    # θs = sqrt.(dfg.var)
    # ss = dfg.mean .- θs
    # ds_total = ShiftedExponential.(ss, θs)             

    # Gamma model
    θs = dfg.var ./ dfg.mean
    αs = dfg.mean ./ θs
    ds_total = Gamma.(αs, θs)    
    return ds_comm, ds_comp, ds_total    
end

"""

Plot the distribution of the mean and variance of the per-worker latency.
"""
function plot_mean_var_distribution(dfg; nflops, nbytes=30048, prune=0.03)

    plt.figure()

    # total latency
    # 2.840789371049846e6
    dfi = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), dfg)

    ## mean cdf
    plt.subplot(3, 3, 1)    

    xs = sort(dfi.mean)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys, label="nbytes: $nbytes")
    # write_table(xs, ys, "cdf_comm_mean_$(nbytes).csv")

    # fitted distribution
    if size(dfi, 1) >= 100
        j1, j2 = ceil(Int, length(xs) * prune), floor(Int, length(xs) * prune)
        d = Distributions.fit(LogNormal, xs[j1:end-j2]) 
        xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.9999)), length=100)
        ys = cdf.(d, xs)
        plt.plot(xs, ys, "k--")
        # write_table(xs, ys, "cdf_comm_mean_fit_$(nbytes).csv")
    end

    plt.ylabel("CDF")
    plt.xlabel("Avg. total latency")

    ## var cdf
    plt.subplot(3, 3, 2)
    xs = sort(dfi.var)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys, label="nbytes: $nbytes")
    # write_table(xs, ys, "cdf_comm_var_$(nbytes).csv")

    # fitted distribution
    if size(dfi, 1) >= 100
        j1, j2 = ceil(Int, length(xs) * prune), floor(Int, length(xs) * prune)
        d = Distributions.fit(LogNormal, xs[j1:end-j2])         
        xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.9999)), length=100)            
        ys = cdf.(d, xs)
        plt.plot(xs, ys, "k--")
        # write_table(xs, ys, "cdf_comm_var_fit_$(nbytes).csv")
    end

    plt.ylabel("CDF")
    plt.xlabel("Total latency var")
    plt.xscale("log")

    # mean-var scatter
    plt.subplot(3, 3, 3)
    xs = dfi.mean
    ys = dfi.var
    plt.plot(xs, ys, ".", label="nbytes: $nbytes")
    # write_table(xs, ys, "scatter_comm_$(nbytes).csv", nsamples=200)

    plt.xlabel("Avg. comm. latency")
    plt.ylabel("Comm. latency var")
    plt.yscale("log")

    return

    # communication latency
    nbytes_all = sort!(unique(dfg.nbytes))

    ## mean cdf
    plt.subplot(3, 3, 4)
    for nbytes in nbytes_all
        dfi = filter(:nbytes => (x)->x==nbytes, dfg)
        xs = sort(dfi.comm_mean)
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, label="nbytes: $nbytes")
        write_table(xs, ys, "cdf_comm_mean_$(nbytes).csv")

        # fitted distribution
        if size(dfi, 1) >= 100
            d = Distributions.fit(LogNormal, xs)            
            xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.9999)), length=100)
            ys = cdf.(d, xs)
            plt.plot(xs, ys, "k--")
            write_table(xs, ys, "cdf_comm_mean_fit_$(nbytes).csv")
        end
    end
    plt.ylabel("CDF")
    plt.xlabel("Avg. comm. latency")

    ## var cdf
    plt.subplot(3, 3, 5)
    for nbytes in nbytes_all
        dfi = filter(:nbytes => (x)->x==nbytes, dfg)
        xs = sort(dfi.comm_var)
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, label="nbytes: $nbytes")
        write_table(xs, ys, "cdf_comm_var_$(nbytes).csv")

        # fitted distribution
        if size(dfi, 1) >= 100
            d = Distributions.fit(LogNormal, xs)
            xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.9999)), length=100)            
            ys = cdf.(d, xs)
            plt.plot(xs, ys, "k--")            
            write_table(xs, ys, "cdf_comm_var_fit_$(nbytes).csv")
        end
    end    
    plt.ylabel("CDF")
    plt.xlabel("Comm. latency var")
    plt.xscale("log")

    # mean-var scatter
    plt.subplot(3, 3, 6)
    for nbytes in nbytes_all
        dfi = filter(:nbytes => (x)->x==nbytes, dfg)
        xs = dfi.comm_mean
        ys = dfi.comm_var
        plt.plot(xs, ys, ".", label="nbytes: $nbytes")
        write_table(xs, ys, "scatter_comm_$(nbytes).csv", nsamples=200)
    end    
    plt.xlabel("Avg. comm. latency")
    plt.ylabel("Comm. latency var")
    plt.yscale("log")    

    # compute latency
    # 2.840789371049846e6
    dfg = filter(:worker_flops => (x)->isapprox(x, 4.54e8, rtol=1e-2), dfg)
    nflops_all = sort!(unique(dfg.worker_flops))

    ## mean cdf
    plt.subplot(3, 3, 7)
    for nflops in nflops_all
        dfi = filter(:worker_flops => (x)->x==nflops, dfg)
        xs = sort(dfi.comp_mean)
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, label="nflops: $(round(nflops, sigdigits=3))")
        write_table(xs, ys, "cdf_comp_mean_$(round(nflops, sigdigits=3)).csv")

        # fitted distribution
        if size(dfi, 1) >= 100
            d = Distributions.fit(LogNormal, xs)
            xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.99999)), length=100)            
            ys = cdf.(d, xs)
            plt.plot(xs, ys, "k--")                        
            write_table(xs, ys, "cdf_comp_mean_fit_$(round(nflops, sigdigits=3)).csv")
        end
    end
    plt.ylabel("CDF")
    plt.xlabel("Avg. comp. latency")
    plt.xscale("log")
    plt.legend()

    ## var cdf
    plt.subplot(3, 3, 8)
    for nflops in nflops_all
        dfi = filter(:worker_flops => (x)->x==nflops, dfg)
        xs = sort(dfi.comp_var)
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, label="nflops: $nflops")
        write_table(xs, ys, "cdf_comp_var_$(round(nflops, sigdigits=3)).csv")

        # fitted distribution
        if size(dfi, 1) >= 100
            i = round(Int, 0.01*length(xs))
            xs = xs[1:end-i]
            d = Distributions.fit(LogNormal, xs)
            xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.99999)), length=100)            
            ys = cdf.(d, xs)
            plt.plot(xs, ys, "k--")
            write_table(xs, ys, "cdf_comp_var_fit_$(round(nflops, sigdigits=3)).csv")
        end
    end    
    plt.ylabel("CDF")
    plt.xlabel("Comp. latency var")
    plt.xscale("log")

    # mean-var scatter
    plt.subplot(3, 3, 9)
    for nflops in nflops_all
        dfi = filter(:worker_flops => (x)->x==nflops, dfg)
        xs = dfi.comp_mean
        ys = dfi.comp_var
        plt.plot(xs, ys, ".", label="nflops: $nflops")
        write_table(xs, ys, "scatter_comp_$(round(nflops, sigdigits=3)).csv", nsamples=200)
    end    
    plt.xlabel("Avg. comp. latency")
    plt.ylabel("Comp. latency var")
    plt.xscale("log")
    plt.yscale("log")        
end

"""

Compute the parameters of the mean and variance meta-distributions associated with the per-worker 
communication and compute latency separately and the correlation between mean and variance.
"""
function copula_df(dfg; prune=0.03, minsamples=101)

    # communication
    comm_df = DataFrame()
    row = Dict{String,Any}()
    nbytes_all = sort!(unique(dfg.nbytes))
    for nbytes in nbytes_all    
        dfi = filter(:nbytes => (x)->x==nbytes, dfg)

        # filter out extreme values
        q1, q2 = quantile(dfi.mean, prune), quantile(dfi.mean, 1-prune)
        dfi = filter(:mean => (x)->q1<=x<=q2, dfi)
        if size(dfi, 1) < minsamples
            continue
        end

        row["nbytes"] = nbytes        
        row["mean_mean"] = mean(dfi.comm_mean)
        row["mean_var"] = var(dfi.comm_mean)
        row["mean_μ"], row["mean_σ"] = params(Distributions.fit(LogNormal, dfi.comm_mean))
        row["var_mean"] = mean(dfi.comm_var)
        row["var_var"] = var(dfi.comm_var)
        row["var_μ"], row["var_σ"] = params(Distributions.fit(LogNormal, dfi.comm_var))        
        row["cor"] = max.(cor(dfi.comm_mean, dfi.comm_var), 0)
        row["nsamples"] = size(dfi, 1)
        push!(comm_df, row, cols=:union)
    end

    # compute
    comp_df = DataFrame()
    row = Dict{String,Any}()
    nflops_all = sort!(unique(dfg.worker_flops))
    for nflops in nflops_all
        dfi = filter(:worker_flops => (x)->x==nflops, dfg)

        # filter out extreme values
        q1, q2 = quantile(dfi.mean, prune), quantile(dfi.mean, 1-prune)
        dfi = filter(:mean => (x)->q1<=x<=q2, dfi)
        if size(dfi, 1) < minsamples
            continue
        end        

        row["nflops"] = nflops
        row["mean_mean"] = mean(dfi.comp_mean)
        row["mean_var"] = var(dfi.comp_mean)
        row["mean_μ"], row["mean_σ"] = params(Distributions.fit(LogNormal, dfi.comp_mean))
        row["var_mean"] = mean(dfi.comp_var)
        row["var_var"] = var(dfi.comp_var)
        row["var_μ"], row["var_σ"] = params(Distributions.fit(LogNormal, dfi.comp_var))        
        row["cor"] = max.(cor(dfi.comp_mean, dfi.comp_var), 0)
        row["nsamples"] = size(dfi, 1)
        push!(comp_df, row, cols=:union)
    end
    comm_df, comp_df
end

"""

Compute the parameters of the mean and variance meta-distributions associated with the total 
per-worker latency and the correlation between mean and variance.
"""
function copula_df_total(dfg; prune=0.03, minsamples=101)
    length(unique(dfg.nbytes)) == 1 || length(unique(dfg.worker_flops)) || error("either nbytes or worker_flops has to be fixed")
    df = DataFrame()
    row = Dict{String,Any}()
    for dfi in groupby(dfg, [:nbytes, :worker_flops])
        nbytes = dfi.nbytes[1]
        worker_flops = dfi.worker_flops[1]

        # filter out extreme values
        q1, q2 = quantile(dfi.mean, prune), quantile(dfi.mean, 1-prune)
        dfi = filter(:mean => (x)->q1<=x<=q2, dfi)
        if size(dfi, 1) < minsamples
            continue
        end

        row["nbytes"] = nbytes
        row["nflops"] = worker_flops
        row["mean_mean"] = mean(dfi.mean)
        row["mean_var"] = var(dfi.mean)
        row["mean_μ"], row["mean_σ"] = params(Distributions.fit(LogNormal, dfi.mean))
        row["var_mean"] = mean(dfi.var)
        row["var_var"] = var(dfi.var)
        row["var_μ"], row["var_σ"] = params(Distributions.fit(LogNormal, dfi.var))        
        row["cor"] = max.(cor(dfi.mean, dfi.var), 0)
        row["nsamples"] = size(dfi, 1)
        push!(df, row, cols=:union)
    end
    df.cor .= mean(filter((x)->x>0, df.cor))
    sort!(df, [:nflops, :nbytes])
    df
end

"""

Interpolate between rows of the DataFrame `df`.
"""
function interpolate_df(dfc, x; key=:nflops)
    sort!(dfc, key)    
    if x in dfc[:, key]
        i = searchsortedfirst(dfc[:, key], x)
        return Dict(pairs(dfc[i, :]))
    end
    size(dfc, 1) > 1 || error("Need at least 2 samples for interpolation")

    # select the two closest points for which there is data
    j = searchsortedfirst(dfc[:, key], x)
    if j > size(dfc, 1)
        j = size(dfc, 1)  
        i = j - 1
    elseif j == 1
        j = 2
        i = 1
    else
        i = j - 1
    end

    # interpolate between, or extrapolate from, those points to x
    rv = Dict{Symbol,Any}()
    for name in names(dfc)
        slope = (dfc[j, name] - dfc[i, name]) / (dfc[j, key] - dfc[i, key])
        intercept = dfc[j, name] - dfc[j, key]*slope
        rv[Symbol(name)] = intercept + slope*x
    end
    rv
end

function sample_worker_distribution(dfc, x; key)
    row = interpolate_df(dfc, x; key)
    
    # mean-distribution
    μ = row[:mean_μ]
    σ = row[:mean_σ]
    d_mean = LogNormal(μ, σ)

    # var-distribution
    μ = row[:var_μ]
    σ = row[:var_σ]
    d_var = LogNormal(μ, σ)

    # copula
    c = row[:cor]
    Σ = [1 c; c 1]
    d = MvNormal(zeros(2), Σ)

    # sample from the copula
    p = rand(d)
    m = quantile(d_mean, cdf(Normal(), p[1]))
    v = quantile(d_var, cdf(Normal(), p[2]))
    
    # worker latency is Gamma-distributed
    θ = v / m
    α = m / θ
    return Gamma(α, θ)

    # # worker latency is ShiftedExponential-distributed
    # θ = sqrt(v)
    # s = m - θ
    # ShiftedExponential(s, θ)
end

sample_worker_comm_distribution(dfc_comm, nbytes) = sample_worker_distribution(dfc_comm, nbytes; key=:nbytes)
sample_worker_comp_distribution(dfc_comp, nflops) = sample_worker_distribution(dfc_comp, nflops; key=:nflops)

"""

Compute all order statistics for `nworkers` workers, when the per-worker 
workload is `worker_flops`, via Monte Carlo sampling over `nsamples` samples.
"""
function predict_latency_niid(nbytes, nflops, nworkers; nsamples=1000, niidm)
    dfc_comm, dfc_comp = niidm
    rv = zeros(nworkers)
    buffer = zeros(nworkers)
    for _ in 1:nsamples
        for i in 1:nworkers
            buffer[i] = 0
            if !isnothing(dfc_comm)
                buffer[i] += rand(sample_worker_comm_distribution(dfc_comm, nbytes))
            end
            if !isnothing(dfc_comp)
                buffer[i] += rand(sample_worker_comp_distribution(dfc_comp, nflops))
            end
        end
        sort!(buffer)
        rv += buffer
    end
    rv ./= nsamples
end

### code for simulating the per-iteration latency

"""

Sample the total latency of a worker.
"""
function sample_worker_latency(d_comm, d_comp)
    !isnothing(d_comm) || !isnothing(d_comp) || error("either d_comm or d_comp must be provided")
    rv = 0.0
    if !isnothing(d_comm)
        rv += rand(d_comm)
    end
    if !isnothing(d_comp)
        rv += rand(d_comp)
    end
    rv
end

"""

Simulate `niterations` iterations of the computation.
"""
function simulate_iterations(;nwait, niterations=100, ds_comm, ds_comp, update_latency=0.5e-3)
    length(ds_comm) == length(ds_comp) || throw(DimensionMismatch("ds_comm has dimension $(length(ds_comm)), but ds_comp has dimension $(length(ds_comp))"))
    nworkers = length(ds_comm)
    0 < nwait <= nworkers || throw(DomainError(nwait, "nwait must be in [1, nworkers]"))
    sepochs = zeros(Int, nworkers) # epoch at which an iterate was last sent to each worker
    repochs = zeros(Int, nworkers) # epoch that each received gradient corresponds to
    stimes = zeros(nworkers) # time at which each worker was most recently assigned a task
    rtimes = zeros(nworkers) # time at which each worker most recently completed a task
    pq = PriorityQueue{Int,Float64}() # queue for event-driven simulation
    time = 0.0 # simulation time
    times = zeros(niterations) # time at which each iteration finished
    nfreshs = zeros(Int, niterations)
    nstales = zeros(Int, niterations)
    latencies = zeros(niterations)
    idle_times = zeros(niterations)
    fresh_times = zeros(niterations)
    stale_times = zeros(niterations)
    for k in 1:niterations
        nfresh, nstale = 0, 0
        idle_time = 0.0 # total time workers spend being idle
        fresh_time = 0.0 # total time workers spend working on fresh gradients
        stale_time = 0.0 # total time workers spend working on stale gradients

        # enqueue all idle workers
        # (workers we received a fresh result from in the previous iteration are idle)
        t0 = time + update_latency # start time of this iteration
        for i in 1:nworkers
            if repochs[i] == k-1
                enqueue!(pq, i, t0 + sample_worker_latency(ds_comm[i], ds_comp[i]))
                sepochs[i] = k
                stimes[i] = t0
            end
        end

        # wait for nwait fresh workers
        while nfresh < nwait
            i, time = dequeue_pair!(pq)
            repochs[i] = sepochs[i]
            rtimes[i] = time
            if k > 1 && time < t0
                idle_time += t0 - time
                time = t0
            end
            if repochs[i] == k
                nfresh += 1
                fresh_time += rtimes[i] - stimes[i]
            else
                # put stale workers back in the queue
                nstale += 1
                stale_time += rtimes[i] - stimes[i]
                enqueue!(pq, i, time + sample_worker_latency(ds_comm[i], ds_comp[i]))
                sepochs[i] = k
                stimes[i] = time
            end
        end

        # tally up for how long each worker has been idle
        # (only workers we received a fresh result from are idle)
        for i in 1:nworkers
            if repochs[i] == k
                idle_time += time - rtimes[i] + update_latency
            end
        end

        # record
        nfreshs[k] = nfresh
        nstales[k] = nstale
        times[k] = time + update_latency
        latencies[k] = time - t0
        idle_times[k] = idle_time
        fresh_times[k] = fresh_time
        stale_times[k] = stale_time
    end
    
    rv = DataFrame()
    rv.time = times
    rv.update_latency = update_latency   
    rv.latency = latencies
    rv.iteration = 1:niterations
    rv.idle_time = idle_times
    rv.fresh_time = fresh_times
    rv.stale_time = stale_times
    rv.nworkers = nworkers
    rv.nwait = nwait
    rv
end

"""

Simulate `niterations` iterations of the computation for `nruns` realizations of the set of workers.
"""
function simulate_iterations(nbytes::Real, nflops::Real; nruns=10, niterations=100, nworkers, nwait, dfc_comm, dfc_comp, update_latency, balanced=false)
    dfs = Vector{DataFrame}()
    for i in 1:nruns
        if !isnothing(dfc_comm)
            if !balanced
                ds_comm = [sample_worker_comm_distribution(dfc_comm, nbytes) for _ in 1:nworkers]
            else
                d = sample_worker_comm_distribution(dfc_comm, nbytes)
                ds_comm = [d for _ in 1:nworkers]
            end
        else
            ds_comm = [nothing for _ in 1:nworkers]
        end
        if !isnothing(dfc_comp)
            if !balanced
                ds_comp = [sample_worker_comp_distribution(dfc_comp, nflops) for _ in 1:nworkers]
            else
                d = sample_worker_comp_distribution(dfc_comp, nflops)
                ds_comp = [d for _ in 1:nworkers]
            end
        else
            ds_comp = [nothing for _ in 1:nworkers]
        end
        df = simulate_iterations(;nwait, niterations, ds_comm, ds_comp, update_latency)
        df.jobid = i
        push!(dfs, df)
    end
    df = vcat(dfs...)
    df = combine(
        groupby(df, :iteration),
        :time => mean => :time,
        :update_latency => mean => :update_latency,
        :latency => mean => :latency,
        :idle_time => mean => :idle_time,
        :fresh_time => mean => :fresh_time,
        :stale_time => mean => :stale_time,
    )
    df.nworkers = nworkers    
    df.nwait = nwait
    df.nbytes = nbytes
    df.worker_flops = nflops
    df
end

"""

"""
function plot_time_vs_npartitions(;nbytes::Real=30048, nflops0::Real=6.545178710898845e10, nworkers, nwait, dfc_comm, dfc_comp, update_latency=0.5e-3)
    # ps = [1, 5, 10, 50, 100, 320]
    ps = 10 .^ range(log10(1), log10(320), length=10) .* nworkers
    idle_times = zeros(length(ps))
    fresh_times = zeros(length(ps))
    stale_times = zeros(length(ps))
    for (i, p) in enumerate(ps)
        df = simulate_iterations(nbytes, nflops0/p; nworkers, nwait, dfc_comm, dfc_comp, update_latency)
        idle_times[i] = mean(df.idle_time)
        fresh_times[i] = mean(df.fresh_time)
        stale_times[i] = mean(df.stale_time)
        total = idle_times[i] + fresh_times[i] + stale_times[i]
        idle_times[i] /= total
        fresh_times[i] /= total
        stale_times[i] /= total
    end
    
    plt.figure()
    plt.plot(ps, idle_times, ".-", label="Idle")
    plt.plot(ps, fresh_times, ".-", label="Fresh")
    plt.plot(ps, stale_times, ".-", label="Stale")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of data partitions")
    plt.ylabel("Time [s]")
    return
end

"""

Plot the average iteration latency (across realizations of the set of workers) vs. the number of workers.
"""
function plot_latency_vs_nworkers(;nbytes::Real=30048, nflops0::Real=6.545178710898845e10/80, ϕ=1, dfc_comm, dfc_comp, update_latency=0.5e-3, df=nothing)

    plt.figure()
    # empirical iteration latency
    if !isnothing(df)
        df = filter([:worker_flops, :nworkers] => (x, y)->isapprox(x*y, nflops0, rtol=1e-2), df)
        df = filter(:nbytes => (x)->x==nbytes, df)
        df = filter([:nwait, :nworkers] => (x, y)->x==round(Int, ϕ*y), df) 
        xs = zeros(Int, 0)
        ys = zeros(0)
        for nworkers in unique(df.nworkers)
            dfi = filter(:nworkers => (x)->x==nworkers, df)
            dfi = combine(groupby(dfi, :jobid), :latency => mean => :latency)
            push!(xs, nworkers)
            push!(ys, mean(dfi.latency))
        end
        plt.plot(xs, ys, "s", label="Empiric")
        write_table(xs, ys, "latency_vs_nworkers_$(round(nflops0, sigdigits=3))_$(round(ϕ, sigdigits=3)).csv")
    end    

    # return

    # simulated iteration latency
    nworkerss = round.(Int, 10 .^ range(log10(10), log10(1000), length=10))
    latencies = zeros(length(nworkerss))
    # for (i, nworkers) in enumerate(nworkerss)
    Threads.@threads for i in 1:length(nworkerss)
        nworkers = nworkerss[i]
        nflops = nflops0 / nworkers
        nwait = max(1, round(Int, ϕ*nworkers))
        println("nworkers: $nworkers, nwait: $nwait, nflops: $(round(nflops, sigdigits=3))")
        df = simulate_iterations(nbytes, nflops; nworkers, nwait, dfc_comm, dfc_comp, update_latency)
        latencies[i] = mean(df.latency)
    end    
    plt.plot(nworkerss, latencies, "-", label="Predicted")
    write_table(nworkerss, latencies, "latency_vs_nworkers_sim_$(round(nflops0, sigdigits=3))_$(round(ϕ, sigdigits=3)).csv")

    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of workers")
    plt.ylabel("Time [s]")
    return
end