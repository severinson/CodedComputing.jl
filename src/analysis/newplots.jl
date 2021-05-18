### latency timeseries

"""

Plot the iteration latency of workers with indices in `workers` of job `jobid`.
"""
function plot_timeseries(df; jobid=rand(unique(df.jobid)), workers=[1, 2], separate=false)
    println("jobid: $jobid")
    df = filter(:jobid => (x)->x==jobid, df)
    plt.figure()
    for worker in workers
        xs = df.iteration        
        if separate
            # compute
            ys = df[:, "compute_latency_worker_$worker"]
            plt.plot(xs, ys, label="Worker $worker (comp.)")
            write_table(xs[1:600], ys[1001:1600], "timeseries_compute_$(jobid)_$(worker).csv", nsamples=600)

            # communication
            ys = df[:, "latency_worker_$worker"] .- df[:, "compute_latency_worker_$worker"]
            plt.plot(xs, ys, label="Worker $worker (comm.)")
            write_table(xs[1:600], ys[1001:1600], "timeseries_communication_$(jobid)_$(worker).csv", nsamples=600)

        end
        ys = df[:, "latency_worker_$worker"]
        plt.plot(xs, ys, label="Worker $worker")
        write_table(xs[1:600], ys[1001:1600], "timeseries_$(jobid)_$(worker).csv", nsamples=600)
        
    end    
    plt.grid()
    plt.legend()
    plt.title("Job $jobid")
    plt.xlabel("Iteration")
    plt.ylabel("Per-worker iteration latency [s]")
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
"""
function plot_prior_orderstats(df; nworkers, nbytes=30048, nflops, iter=10, niidm=nothing)
    df = filter(:nworkers => (x)->x==nworkers, df)
    df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    df = filter(:nbytes => (x)->x==nbytes, df)    
    df = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), df)
    if size(df, 1) == 0
        error("no rows match nbytes: $nbytes and nflops: $nflops")
    end
    latency_columns = ["latency_worker_$(i)" for i in 1:maximum(df.nworkers)]
    buffer = zeros(nworkers)
    orderstats = zeros(nworkers)
    jobids = unique(df.jobid)
    println("Computing orderstats over $(length(jobids)) jobs")
    for jobid in jobids
        dfi = filter(:jobid=>(x)->x==jobid, df)
        sort!(dfi, :iteration)
        for j in 1:nworkers
            buffer[j] = dfi[iter, latency_columns[j]]
        end
        sort!(buffer)
        orderstats += buffer
    end    
    orderstats ./= length(jobids)
    xs = 1:nworkers
    plt.figure()
    plt.plot(xs, orderstats, "-o")
    write_table(xs, orderstats, "prior_orderstats_$(iter)_$(nworkers)_$(nbytes)_$(round(nflops, sigdigits=3)).csv")

    # latency predicted by the new non-iid model
    if !isnothing(niidm)
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

### non-iid model

"""

Plot the latency distribution of individual workers.
"""
function plot_worker_latency_distribution(df; jobid=rand(unique(df.jobid)), worker_indices=[1, 10])
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
        write_table(xs, ys, "./results/cdf_$(jobid)_$(i).csv")
        j = 100
        d = Distributions.fit(Gamma, xs[1:end-j])
        xs = range(quantile(d, 0.000001), quantile(d, 0.999999), length=100)
        ys = cdf.(d, xs)
        write_table(xs, ys, "./results/cdf_fit_$(jobid)_$(i).csv")
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

    # communication    
    plt.subplot(3, 1, 2)
    for i in worker_indices        
        xs = df[:, "latency_worker_$(i)"] .- df[:, "compute_latency_worker_$(i)"]
        sort!(xs)
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, label="Worker $i")
        write_table(xs, ys, "./results/cdf_communication_$(jobid)_$(i).csv")
        # plt.hist(xs, 200, density=true)
        # j = round(Int, 0*length(xs))
        j = 100
        d = Distributions.fit(Gamma, xs[1:end-j])
        xs = range(quantile(d, 0.000001), quantile(d, 0.999999), length=100)
        ys = cdf.(d, xs)
        write_table(xs, ys, "./results/cdf_fit_compute_$(jobid)_$(i).csv")
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
        write_table(xs, ys, "./results/cdf_compute_$(jobid)_$(i).csv")
        # j = round(Int, 0.01*length(xs))
        j = 0
        d = Distributions.fit(Gamma, xs[1:end-j])
        xs = range(quantile(d, 0.000001), quantile(d, 0.999999), length=100)
        ys = cdf.(d, xs)
        write_table(xs, ys, "./results/cdf_fit_communication_$(jobid)_$(i).csv")
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
function worker_distribution_df(df; minsamples=100, prune_comm=0.05)
    df = filter([:nwait, :nworkers] => (x,y)->x==y, df)
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
                row["comp_mean"] = mean(ys)
                row["comp_var"] = var(ys)

                # communication latency
                ys = float.(dfi[:, latency_columns[i]] .- dfi[:, compute_latency_columns[i]])
                if prune_comm != zero(prune_comm)
                    j = round(Int, prune_comm*length(ys))
                    sort!(ys)
                    ys = ys[1:end-j]
                end                
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
            row["mean"] = mean(ys)
            row["var"] = var(ys)

            # Gamma
            d = Distributions.fit(Gamma, ys)     
            row["α"], row["θ"] = params(d)            
            
            # ShiftedExponential
            d = Distributions.fit(ShiftedExponential, float.(dfi[:, latency_columns[i]]))
            row["s"], row["sθ"] = params(d)

            row["worker_index"] = i
            push!(rv, row, cols=:union)
        end
    end
    rv
end

"""

Plot the distribution of the mean and variance of the per-worker latency.
"""
function plot_mean_var_distribution(dfg)

    plt.figure()

    # communication latency
    nbytes_all = sort!(unique(dfg.nbytes))

    ## mean cdf
    plt.subplot(2, 3, 1)
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
    plt.subplot(2, 3, 2)
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
    plt.subplot(2, 3, 3)
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
    dfg = filter(:worker_flops => (x)->isapprox(x, 2.840789371049846e6, rtol=1e-2), dfg)
    nflops_all = sort!(unique(dfg.worker_flops))

    ## mean cdf
    plt.subplot(2, 3, 4)
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
    plt.subplot(2, 3, 5)
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
    plt.subplot(2, 3, 6)
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

Compute the parameters of the mean and variance meta-distributions, 
and the correlation between mean and variance.
"""
function copula_df(dfg)

    # communication
    comm_df = DataFrame()
    row = Dict{String,Any}()
    nbytes_all = sort!(unique(dfg.nbytes))
    for nbytes in nbytes_all    
        dfi = filter(:nbytes => (x)->x==nbytes, dfg)
        row["nbytes"] = nbytes        
        row["mean_mean"] = mean(dfi.comm_mean)
        row["mean_var"] = var(dfi.comm_mean)
        row["mean_μ"], row["mean_σ"] = params(Distributions.fit(LogNormal, dfi.comm_mean))
        row["var_mean"] = mean(dfi.comm_var)
        row["var_var"] = var(dfi.comm_var)
        row["var_μ"], row["var_σ"] = params(Distributions.fit(LogNormal, dfi.comm_var))        
        row["cor"] = cor(dfi.comm_mean, dfi.comm_var)
        push!(comm_df, row, cols=:union)
    end

    # compute
    comp_df = DataFrame()
    row = Dict{String,Any}()
    nflops_all = sort!(unique(dfg.worker_flops))
    for nflops in nflops_all
        dfi = filter(:worker_flops => (x)->x==nflops, dfg)
        row["nflops"] = nflops
        row["mean_mean"] = mean(dfi.comp_mean)
        row["mean_var"] = var(dfi.comp_mean)
        row["mean_μ"], row["mean_σ"] = params(Distributions.fit(LogNormal, dfi.comp_mean))
        row["var_mean"] = mean(dfi.comp_var)
        row["var_var"] = var(dfi.comp_var)
        row["var_μ"], row["var_σ"] = params(Distributions.fit(LogNormal, dfi.comp_var))        
        row["cor"] = cor(dfi.comp_mean, dfi.comp_var)
        push!(comp_df, row, cols=:union)
    end
    comm_df, comp_df
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
    
    # # worker latency is Gamma-distributed
    # θ = v / m
    # α = m / θ
    # return Gamma(α, θ)

    # worker latency is ShiftedExponential-distributed
    θ = sqrt(v)
    s = m - θ
    ShiftedExponential(s, θ)
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
            buffer[i] = rand(sample_worker_comm_distribution(dfc_comm, nbytes))
            buffer[i] += rand(sample_worker_comp_distribution(dfc_comp, nflops))
        end
        sort!(buffer)
        rv += buffer
    end
    rv ./= nsamples
end

### old code associated with the non-iid model

"""

Show how the avg. per-worker latency scales with nflops, and the distribution of the avg. 
per-worker latency.
"""
function plot_gamma_mean_distribution(dfg)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(dfg.worker_flops, dfg.mean, ".")
    plt.xlabel("Workload [flops]")
    plt.ylabel("Avg. per-worker latency")
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Avg. latency per worker")

    plt.subplot(1, 2, 2)
    for worker_flops in sort!(unique(dfg.worker_flops))
        dfi = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), dfg)
        xs = sort(dfi.mean)
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, 1 .- ys)
        
        i = round(Int, 0.01*length(xs))
        # println("samples: $(size(dfi, 1)), i: $i")
        xs = xs[i:end-i]
        d = Distributions.fit(Gamma, xs)
        xs = range(quantile(d, 0.001), quantile(d, 0.999), length=100)
        plt.plot(xs, 1 .- cdf.(d, xs), "k--")
    end
    # plt.ylim(1e-2, 1)
    plt.xlabel("Avg. per-worker latency")
    plt.ylabel("CCDF")    
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.title("CCDF of avg. latency 
    for each workload.")
    plt.tight_layout()
    return
end

"""

Fit a probability distribution to the avg. per-worker latency for each value of 
`worker_flops`.
"""
function gamma_mean_df(dfg; minsamples=100)
    rv = DataFrame()
    row = Dict{String, Any}()    
    for worker_flops in unique(dfg.worker_flops)
        dfi = filter(:worker_flops => (x)->isapprox(x, worker_flops, rtol=1e-2), dfg)
        nsamples = size(dfi, 1)
        if nsamples < minsamples
            continue
        end

        # avg. per-worker latency is Gamma
        xs = sort(dfi.mean)
        i = round(Int, 0.01*length(xs))
        xs = xs[i:end-i]
        d = Distributions.fit(Gamma, xs)
        α, θ = params(d)
        row["α"] = α
        row["θ"] = θ
        row["meta_mean"], row["meta_var"] = α*θ, α*θ^2

        # avg. per-worker latency is ShiftedExponential
        xs = sort(dfi.mean)
        # i = round(Int, 0.05*length(xs))
        # xs = xs[i:end-i]
        d = Distributions.fit(ShiftedExponential, xs)
        s, θ = params(d)
        row["s"] = s
        row["sθ"] = θ

        row["worker_flops"] = worker_flops
        row["nbytes"] = dfi.nbytes[1]
        row["nsamples"] = nsamples
        push!(rv, row, cols=:union)
    end
    rv
end

"""

Plot the parameters of the distribution fit to the mean latency vs. nflops.
"""
function plot_gamma_mean_df(dfgm)
    plt.figure()

    # Gamma model
    plt.subplot(1, 3, 1)
    xs = dfgm.worker_flops
    ys = dfgm.meta_mean
    plt.plot(xs, ys, "b.", label="Meta-mean")

    ts = sort(xs)
    p, coeffs = fit_polynomial(xs, ys, 1)
    plt.plot(ts, p.(ts), "b--")
    println("meta-mean coefficients: $coeffs")    

    # ShiftedExponential model
    xs = dfgm.worker_flops
    ys = dfgm.s
    plt.plot(xs, ys, "r.", label="Meta-shift")

    ts = sort(xs)
    p, coeffs = fit_polynomial(xs, ys, 1)
    plt.plot(ts, p.(ts), "r--")
    println("meta-shift coefficients: $coeffs")

    plt.ylabel("meta-mean and meta-shift")    
    plt.xlabel("workload [flops]")
    plt.legend()    
    plt.xscale("log")
    plt.yscale("log")

    # variance vs. workload
    plt.subplot(1, 3, 2)
    xs = dfgm.worker_flops
    # ys = sqrt.(dfgm.var_mean)
    ys = sqrt.(dfgm.meta_var)
    plt.plot(xs, ys, ".")

    ts = sort(xs)
    p, coeffs = fit_polynomial(xs, ys, 1)
    plt.plot(ts, p.(ts), "k--")
    println("meta-var coefficients: $coeffs")    

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("workload")
    plt.ylabel("meta-variance")
    

    # scale vs. nflops
    ## Gamma model
    plt.subplot(1, 3, 3)
    ys = dfgm.θ
    plt.plot(xs, ys, "b.", label="Meta-scale (Gamma)")
    
    p, coeffs = fit_polynomial(xs, ys, 1)
    plt.plot(ts, p.(ts), "k--")    
    println("Gamma scale coefficients: $coeffs")

    ## ShiftedExponential model
    ys = dfgm.sθ
    plt.plot(xs, ys, "r.", label="Meta-scale (ShiftedExponential)")
    
    p, coeffs = fit_polynomial(xs, ys, 1)
    plt.plot(ts, p.(ts), "r--")    
    println("ShiftedExponential scale coefficients: $coeffs")    

    plt.xlabel("workload [flops]")
    plt.ylabel("meta-scale")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")        
    plt.tight_layout()
    return
end

"""

Plot the distribution of the normalized variance.
"""
function plot_gamma_var_distribution(dfg)

    plt.figure()
    plt.subplot(1, 2, 1)
    xs = dfg.mean
    ys = dfg.θ # equal to dfg.var ./ dfg.mean
    plt.plot(xs, ys, ".")
    plt.xlabel("avg. per-worker latency")
    # plt.ylabel("per-worker latency var. / avg. per-worker latency")
    plt.ylabel("scale (θ)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()

    plt.subplot(1, 2, 2)
    xs = sort(ys)
    i = round(Int, 0.05*length(xs))
    xs = xs[1:end-i]
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys, label="Empirical")

    d = Distributions.fit(Exponential, xs)
    println(d)
    ts = range(quantile(d, 0.0001), quantile(d, 0.9999), length=100)
    plt.plot(ts, cdf.(d, ts), "k--", label="Fitted Exponential distribution")
    plt.ylabel("CCDF")
    # plt.xlabel("per-worker latency var. / avg. per-worker latency")
    plt.xlabel("scale (θ)")
    plt.ylim(1e-2, 1)
    # plt.xscale("log")
    # plt.yscale("log")    
    plt.legend()
    plt.grid()
    plt.tight_layout()
    return
end

function fit_non_iid_model(dfg, dfgm)

    # interpolate the mean and variance of the meta distribution
    xs = dfgm.worker_flops
    ys = dfgm.meta_mean
    # p_mean, coeffs = fit_polynomial(xs, ys, 1)
    # println("meta-mean coefficients: $coeffs")
    mean_slope = mean(ys ./ xs)
    p_mean = (x) -> x*mean_slope
    println("meta-mean slope: $mean_slope")
    
    ys = dfgm.meta_var
    # p_var, coeffs = fit_polynomial(xs, ys, 1)
    # println("meta-var coefficients: $coeffs")
    var_slope = mean(ys ./ xs)
    p_var = (x) -> x*var_slope
    println("meta-var slope: $var_slope")

    # # fit the meta-distribution determining avg. latency
    # xs = dfgm.worker_flops
    # ys = dfgm.s
    # p_shift, coeffs = fit_polynomial(xs, ys, 1)
    # println("meta-shift coefficients: $coeffs")
    # ys = dfgm.sθ
    # p_scale, coeffs = fit_polynomial(xs, ys, 1)
    # println("meta-scale coefficients: $coeffs")

    # fit the scale distribution (i.e., the distribution of meta_mean / meta_var)
    ys = sort(dfg.θ)
    i = round(Int, 0.05*length(ys))
    ys = ys[1:end-i]
    d = Distributions.fit(Exponential, ys)

    (p_mean, p_var, d)
end

"""

Generate a `ShiftedExponential` random variable for a worker with given per-worker workload
"""
function shiftexp_worker_distribution(worker_flops; osm)
    p_shift, p_scale, dθ = osm
    meta_shift = p_shift(worker_flops)
    meta_scale = p_scale(worker_flops)    
    d = ShiftedExponential(meta_shift, meta_scale)
    mean = rand(d) # avg. latency of this worker, equal to s+sθ
    θ = rand(dθ)
    
    # worker latency distribution is a ShiftedExponential
    # sθ = sqrt(θ * mean)
    # s = mean - sθ    
    # ShiftedExponential(s, sθ)

    # worker latency distribution is a Gamma
    α = mean / θ
    Gamma(α, θ)
end

"""

Generate a `Gamma` random variable for a worker with given per-worker workload
"""
function gamma_worker_distribution(worker_flops; osm)

    p_mean, p_var, scale_distribution = osm
    meta_mean = p_mean(worker_flops)
    meta_var = p_var(worker_flops)
    meta_scale = sqrt(meta_var)
    meta_shift = meta_mean - meta_scale
    d = ShiftedExponential(meta_shift, meta_scale)
    mean = rand(d)
    θ = rand(scale_distribution)
    α = mean / θ
    # println("workload: $worker_flops, meta_mean: $meta_mean, meta_var: $meta_var, α: $α, θ: $θ")
    return Gamma(α, θ)

    # p_shift, p_scale, dθ = osm
    # meta_shift = p_shift(worker_flops)
    # meta_scale = p_scale(worker_flops)
    # d = ShiftedExponential(meta_shift, meta_scale)
    # mean = rand(d) # avg. latency of this worker, equal to α*θ    
    # θ = rand(dθ)
    # α = mean / θ
    # return Gamma(α, θ)


    # # ShiftedExponential model
    # # generate mean latency of the worker based on the per-worker workload
    # meta_shift = 0.0007499872578485828 + 7.778654414066668e-9worker_flops
    # meta_scale = -8.190460220896224e-5 + 5.58026508893015e-10worker_flops
    # d = ShiftedExponential(meta_shift, meta_scale)
    # mean = rand(d) # avg. latency of this worker

    # # Gamma model
    # meta_mean = 0.0006680826556396607 + 8.336680922959681e-9worker_flops
    # meta_scale = -1.2720269369198488e-5 + 3.5940299549679294e-11worker_flops
    # meta_shape = meta_mean / meta_scale
    # d = Gamma(meta_shape, meta_scale)
    # mean = rand(d)

    # # Gamma model w. removing 5% largest and smallest values
    # meta_mean = 0.0006275309187944665 + 8.322795865015209e-9worker_flops
    # meta_scale = -1.1809861023747872e-5 + 2.936533045933439e-11worker_flops
    # meta_shape = meta_mean / meta_scale
    # d = Gamma(meta_shape, meta_scale)
    # mean = rand(d)

    # Shifted exponential w. removing 5% largest and smallest values
    meta_shift = 0.0007199382199680563 + 7.81714364759284e-9worker_flops
    meta_scale = -9.240730117362186e-5 + 5.056522174223721e-10worker_flops
    d = ShiftedExponential(meta_shift, meta_scale)
    mean = rand(d) # avg. latency of this worker, equal to α*θ

    # Gamma scale
    # θ = rand(Gamma(16.942689050787024, 3.946363854792131e-6))

    # Normal-distribution scale
    # dθ = Normal(6.686201567300831e-5, 1.5062764605542296e-5)
    # θ = rand(dθ)
    # while θ <= 0
    #     θ = rand(dθ)
    # end

    # ShiftedExponential scale
    # θ = rand(ShiftedExponential(5.592635406074897e-7, 6.668430692272587e-6))

    ## LogNormal scale
    θ = rand(LogNormal(-12.128750442775992, 0.7232943784486586))
    
    # shape parameter is computed from the above
    α = mean / θ
    return Gamma(α, θ)
end

"""

Compute all order statistics for `nworkers` workers, when the per-worker 
workload is `worker_flops`, via Monte Carlo sampling over `nsamples` samples.
"""
function predict_latency_gamma(worker_flops, nworkers; nsamples=1000, osm)
    rv = zeros(nworkers)
    buffer = zeros(nworkers)
    for _ in 1:nsamples
        for i in 1:nworkers
            d = gamma_worker_distribution(worker_flops; osm)
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