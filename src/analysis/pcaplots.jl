"""

Divide the 
"""
function split_df_by_time(df, window)
    length(unique(df.jobid)) == 1 || throw(ArgumentError("DF may only contain 1 job"))
    is = ceil.(Int, df.t_total ./ window)
    [df[is .== i, :] for i in (1:is[end])]
end

"""

"""
function plot_worker_latency_quantiles(df; nworkers=18, nsubpartitions=4, window=10, qs=[0.1, 0.5, 0.9])
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.pfraction .== 1, :]    

    df = df[df.nworkers .== nworkers, :]
    df = df[df.nsubpartitions .== nsubpartitions, :]
    sort!(df, [:jobid, :iteration])

    nwait = 9
    df = df[df.nwait .== nwait, :]

    samples = Matrix{Union{Float64,Missing}}(undef, size(df, 1), nworkers)

    jobids = unique(df.jobid)
    println("Computing staleness over $(length(jobids)) jobs")
    j = 1
    for jobid in jobids

        dfi = df
        dfi = dfi[dfi.jobid .== jobid, :]
        dfs = split_df_by_time(dfi, window)
        # println(length(dfs))
        # continue

        # if jobid == 318
        #     return dfs
        # end

        for dfj in dfs
            if size(dfj, 1) == 0
                continue
            end

            M  = staleness_matrix_from_jobid(dfj, jobid, nworkers)        

            # sort the workers by average latency within the time window
            means = [mean(skipmissing(M[:, i])) for i in 1:nworkers]
            p = sortperm(means)
            # println(means)
            # println(jobid)
            # println()

            # record latency
            for i in 1:size(M, 1) # iteration
                samples[j, :] .= M[i, p]
                j += 1
            end            
        end
    end   
    @assert j == size(samples, 1)  + 1

    # plot ccdf
    plt.figure()    

    # all workers
    ss = collect(skipmissing(vcat([view(samples, :, i) for i in 1:nworkers]...)))
    sort!(ss)
    ys = 1 .- (0:length(ss)-1) ./ (length(ss)-1)
    # ccdf = 1 .- cdf    
    plt.semilogy(ss, ys, label="All workers") 

    # fit a Gamma
    # shift = minimum(ss) - eps(Float64)
    shift = 0    
    ss .-= shift
    e = Distributions.fit(Gamma, ss)
    ts = range(0, maximum(ss), length=100)
    ys = 1 .- Distributions.cdf.(e, ts)
    plt.semilogy(ts .+ shift, ys)
    println(params(e))

    # scale = μ / sum(1/i for i in (nworkers-nwait+1):nworkers)    
    # # get the scale from waiting for all workers
    # β = 0.0        
    # for nworkers in unique(df.nworkers)
    #     dfi = df
    #     dfi = dfi[dfi.nworkers .== nworkers, :]
    #     nwait = nworkers
    #     ts = dfi[dfi.nwait .== nwait, :t_compute] .- shift
    #     # σ = var(ts)
    #     # β1 = sqrt(σ / sum(1/i^2 for i in (nworkers-nwait+1):nworkers))        
    #     μ = mean(ts)
    #     βi = μ / sum(1/i for i in (nworkers-nwait+1):nworkers)
    #     β += βi * size(dfi, 1) / size(df, 1)
    # end
    # return shift, β    

    # fast workers
    ss = collect(skipmissing(vcat([view(samples, :, i) for i in 1:9]...)))
    sort!(ss)
    cdf = (0:length(ss)-1) ./ (length(ss)-1)
    ccdf = 1 .- cdf    
    plt.semilogy(ss, ccdf, label="Fastest 50%")

    # fit a Gamma
    # shift = minimum(ss) - eps(Float64)
    shift = 0    
    ss .-= shift
    e = Distributions.fit(Gamma, ss)
    ts = range(0, maximum(ss), length=100)
    ys = 1 .- Distributions.cdf.(e, ts)
    plt.semilogy(ts .+ shift, ys)
    println(params(e))    

    # # fit a shifted exponential    
    # shift = quantile(ss, 0.01)
    # ss .-= shift
    # scale = mean(ss)
    # e = Exponential(scale)
    # ts = range(minimum(ss), maximum(ss), length=100)
    # ccdf = 1 .- Distributions.cdf.(e, ts)
    # plt.semilogy(ts.+shift, ccdf)    

    # slow workers
    ss = collect(skipmissing(vcat([view(samples, :, i) for i in 9:18]...)))
    sort!(ss)
    cdf = (0:length(ss)-1) ./ (length(ss)-1)
    ccdf = 1 .- cdf    
    plt.semilogy(ss, ccdf, label="Slowest 50%")

    # fit a Gamma
    # shift = minimum(ss) - eps(Float64)
    shift = 0
    ss .-= shift
    e = Distributions.fit(Gamma, ss)
    ts = range(0, maximum(ss), length=100)
    ys = 1 .- Distributions.cdf.(e, ts)
    plt.semilogy(ts .+ shift, ys)    
    println(params(e))    

    # # fit a shifted exponential    
    # shift = quantile(ss, 0.01)
    # ss .-= shift
    # scale = mean(ss)
    # e = Exponential(scale)
    # ts = range(minimum(ss), maximum(ss), length=100)
    # ccdf = 1 .- Distributions.cdf.(e, ts)
    # plt.semilogy(ts.+shift, ccdf)    
    
    # samples_fast = vcat([view(samples, :, i) for i in 1:9]...)
    # samples_slow = vcat([view(samples, :, i) for i in 9:18]...)


    # plt.figure()
    # for i in 1:nworkers
    #     vs = collect(skipmissing(view(samples, :, i)))
    #     sort!(vs)
    #     cdf = (0:length(vs)-1) ./ (length(vs)-1)
    #     ccdf = 1 .- cdf
    #     plt.semilogy(vs, ccdf, label="$(i)-th fastest worker")        
    # end

    plt.grid()
    plt.legend()
    plt.xlabel("Compute latency [s]")
    plt.ylabel("CCDF")
    plt.ylim(1e-2, 1)
    return

    # plot quantiles
    plt.figure()
    for q in qs
        plt.plot(1:nworkers, [quantile(skipmissing(view(samples, :, i)), q) for i in 1:nworkers], "o", label="q: $q")
    end
    plt.legend()
    plt.xlabel("Workers")
    plt.ylabel("Latency [s]")
    plt.grid()
    return
end

