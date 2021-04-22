# Code for analyzing and plotting the correlation of compute latency

"""

Return a matrix of size `niterations` by `nworkers`, where a `1` in position `i, j` indicates that
worker `j` was a straggler in iteration `i`. Non-stragglers are marked with `-1`.
"""
function straggler_matrix_from_jobid(df, jobid, nworkers)
    df = filter(:jobid => (x)->x==jobid, df)
    sort!(df, :iteration)
    rv = fill(-1, size(df, 1), nworkers)
    for i in 1:nworkers
        rv[df["repoch_worker_$(i)"] .< df.iteration, i] .= 1
    end
    ts = zeros(size(df, 1))
    ts[2:end] .= df.time[1:end-1]
    rv, ts
end

"""

Return a matrix of size `niterations` by `nworkers`, where `i, j`-th entry is the time worker `j` 
required to compute the result received by the coordinator in uteration `i`. If `time` is false,
the entry is the number of iterations that has passed instead.
"""
function staleness_matrix_from_jobid(df, jobid, nworkers; time=true)
    df = df[df.jobid .== jobid, :]
    sort!(df, :iteration)    
    rv = Matrix{Union{Float64,Missing}}(undef, size(df, 1), nworkers)
    rv .= missing
    epoch0 = minimum(df.iteration) - 1 # first epoch of this df
    for i in 1:nworkers
        for j in 1:size(df, 1)            
            repoch = df[j, "repoch_worker_$(i)"]
            if repoch <= epoch0 || (j > 1 && repoch == df[j-1, "repoch_worker_$(i)"]) # didn't receive anything this epoch
                continue
            end
            repoch -= epoch0
            if time
                @views rv[j, i] = sum(df.t_compute[repoch:j]) + sum(df.t_update[repoch:(j-1)])
            else
                rv[j, i] = j - repoch
            end
        end
    end
    rv
end


"""

Plot the CCDF of how stale results are, as measured by the number of iterations that has passed.
"""
function plot_staleness(df, nworkers=36, nwait=3, nsubpartitions=10)
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.nworkers .== nworkers, :]
    df = df[df.nwait .== nwait, :]  
    df = df[df.nsubpartitions .== nsubpartitions, :]

    # let's assume results are never more than 10 iterations stale    
    edges = 0:9
    values = zeros(10) 
    count = 0

    jobids = unique(df.jobid)
    for jobid in jobids
        dfi = df
        dfi = dfi[dfi.jobid .== jobid, :]
        if ismissing(dfi.repoch_worker_1[1])
            continue
        end
        if ismissing(dfi.mse[1])
            continue
        end
        println(jobid)
        M = staleness_matrix_from_jobid(dfi, jobid, nworkers, time=false)
        for i in 1:size(M, 1)
            for j in 1:size(M, 2)
                if ismissing(M[i, j])
                    continue
                end
                v = Int(M[i, j])
                values[v+1] += 1
                count += 1
            end
        end
    end
    values ./= count
    println(values)
    println(count)

    # plt.figure()
    cdf = cumsum(values)

    for i in 1:length(cdf)
        println("$(edges[i]) $(cdf[i])")
    end

    plt.plot(edges, cdf)
    return
end


"""

Return the empirical probability of workers still being straggler after some time has passed.
"""
function straggler_prob_timeseries_from_df(df; nbins=nothing, prob=true)

    # histogram bins
    if isnothing(nbins) # bin by iteration
        edges = 0:maximum(df.iteration)
        values = zeros(Union{Missing,Float64}, maximum(df.iteration))
    else # bin by time
        edges = range(0, maximum(df.t_total), length=nbins+1)
        values = zeros(Union{Missing,Float64}, nbins)
    end
    counts = zeros(Int, length(values))

    # average iteration length
    t_totals = zeros(maximum(df.iteration))
    t_total_counts = zeros(Int, length(t_totals))

    jobids = unique(df.jobid)
    println("Computing AC over $(length(jobids)) jobs")
    for jobid in jobids
        dfi = df
        dfi = dfi[dfi.jobid .== jobid, :]
        if ismissing(dfi.repoch_worker_1[1])
            continue
        end
        @assert length(unique(dfi.nworkers)) == 1
        nworkers = unique(dfi.nworkers)[1]

        M, ts = straggler_matrix_from_jobid(df, jobid, nworkers)
        # println("jobid: $jobid, iterations: $(size(M, 1))")

        # autocorrelation
        C = StatsBase.autocor(M, 0:size(M, 1)-1, demean=false)

        # fix normalization made by the autocor function        
        for i in 1:size(C, 1)
            C[i, :] ./= (size(C, 1) - i + 1) / size(C, 1)
        end

        # convert to probability
        if prob
            C .+= 1
            C ./= 2
        end

        # bin the values
        if isnothing(nbins)
            for i in 1:size(C, 1)
                values[i] += sum(view(C, i, :))
                counts[i] += size(C, 2)
            end
        else
            for i in 1:size(C, 1)
                j = searchsortedfirst(edges, ts[i])
                if j > nbins
                    continue
                end            
                j = max(j, 1)
                values[j] += sum(view(C, i, :))
                counts[j] += size(C, 2)
            end
        end
    end
    values ./= counts
    return edges, values, counts
end

"""

Plot the probability that a straggler remains a straggler after some time has passed.
"""
function plot_straggler_ac(df; phi=0.5)
    df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    df = filter(:nreplicas => (x)->x==1, df)

    # df = df[df.nworkers .== nworkers, :]
    # df = df[df.nwait .== nwait, :]


    # df = df[df.nsubpartitions .== 4, :]



    # plot for different subpartitions
    # plt.figure()        
    # nworkers = 12
    # nbins = 50
    # for nsubpartitions in [1, 2, 3, 4, 5]        
    #     dfi = df
    #     dfi = dfi[dfi.nworkers .== nworkers, :]        
    #     dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
    #     if size(dfi, 1) == 0
    #         continue
    #     end
    #     edges, values, counts = straggler_prob_timeseries_from_df(dfi; nbins)
    #     is = counts .>= 10
    #     plt.plot(edges[1:end-1][is], values[is], label="Np: $nsubpartitions")        

    #     # print values
    #     println("Np: $nsubpartitions")
    #     for i in 1:length(values)
    #         println("$(edges[i]) $(values[i])")
    #     end        
    # end
    # plt.legend()
    # plt.grid()
    # plt.xlabel("Time passed [s]")
    # plt.ylabel("Prob. still a straggler")
    # # plt.ylim(0.75, 1.0)
    # plt.xlim(0, 250)    

    # # plot for different nworkers at the same workload
    # plt.figure()    
    # for (nworkers, nsubpartitions, nbins) in zip([6, 12, 18], [6, 3, 2], [30, 30, 60])
    #     nwait = round(Int, nworkers*f)
    #     dfi = df
    #     dfi = dfi[dfi.nworkers .== nworkers, :]
    #     dfi = dfi[dfi.nwait .== nwait, :]
    #     dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
    #     if size(dfi, 1) == 0
    #         continue
    #     end        
    #     # nbins = nothing
    #     edges, values, counts = straggler_prob_timeseries_from_df(dfi; nbins)
    #     plt.plot(edges[1:end-1], values, label="Nn: $nworkers")

    #     # print values
    #     # println("Nn: $nworkers")
    #     # for i in 1:length(values)
    #     #     println("$(edges[i]) $(values[i])")
    #     # end
    # end    
    # plt.legend()
    # plt.grid()
    # plt.xlabel("Time passed [s]")
    # plt.ylabel("Prob. still a straggler")
    # # plt.ylim(0.75, 1.0)
    # # plt.xlim(0, 250)    

    # plot for different nworkers
    plt.figure()    
    for nworkers in [36, 72, 108]
        nwait = round(Int, phi*nworkers)
        dfi = df
        dfi = filter(:nworkers => (x)->x==nworkers, dfi)
        if size(dfi, 1) == 0
            continue
        end
        nbins = round(Int, maximum(dfi.t_total) / 10)
        # return straggler_prob_timeseries_from_df(dfi; nbins)
        edges, values, counts = straggler_prob_timeseries_from_df(dfi; nbins)
        plt.plot(edges[1:end-1], values, ".-", label="$nworkers workers")

        # # print values
        # println("Nn: $nworkers")
        # for i in 1:length(values)
        #     println("$(edges[i]) $(values[i])")
        # end
    end    

    plt.legend()
    plt.grid()
    plt.xlabel("Time passed [s]")
    plt.ylabel("Prob. still a straggler")
    # plt.ylim(0.75, 1.0)
    plt.xlim(0, 250)
    return
end

"""

For a given job id, plot traces indicating which workers responded in each iteration.
"""
function plot_stragglers(df; jobid=362)
    if isnothing(jobid) || !(jobid in df.jobid)
        println("jobid must be one of:")
        println(unique(df.jobid))
        return
    end
    df = df[df.jobid .== jobid, :]
    nworkers = df.nworkers[1]
    # niterations = maximum(df.nworkers)
    plt.figure()
    for i in 1:nworkers
        x = findall(df["repoch_worker_$(i)"] .< df.iteration)
        y = repeat([i], length(x))
        plt.plot(x, y, "ro")

        # print values
        for i in 1:length(x)
            println("$(x[i]) $(y[i])")
        end
    end
    println("Total time: $(maximum(df.t_total))")
    plt.title("Markers indicate which workers are stragglers")
    plt.ylabel("Worker index")
    plt.xlabel("Iteration")
    plt.xlim(0, size(df, 1))
    plt.ylim(0, nworkers)
    plt.grid()
end

"""

For a given job id, plot the fraction of iterations that each worker has responded in against the
iteration index.
"""
function plot_response_fraction(df; jobid=nothing)
    if isnothing(jobid) || !(jobid in df.jobid)
        println("jobid must be one of:")
        println(unique(df.jobid))
        return
    end
    # df = df[df.iteration .>= 10, :]
    df = df[df.jobid .== jobid, :]
    nworkers = df.nworkers[1]
    nwait = df.nwait[1]
    niterations = size(df, 1)
    plt.figure()
    for i in 1:nworkers
        p = sortperm(df.iteration)
        y = cumsum(df["worker_$(i)_responded"][p]) ./ (1:niterations)
        x = 1:niterations
        plt.plot(x, y, label="Worker $i")
    end
    plt.grid()
    plt.legend()
    plt.xlim(0, niterations)
    plt.ylim(0, 1)
    plt.xlabel("Iteration")
    plt.ylabel("Fraction of iterations participated in")        
    plt.title("nworkers: $nworkers, nwait: $nwait")
end

"""

Plot a histogram over how stale updates are when received by the coordinator.
"""
function plot_worker_latency_cdf(df; nworkers=18, nsubpartitions=4)
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.pfraction .== 1, :]    

    df = df[df.nworkers .== nworkers, :]
    df = df[df.nsubpartitions .== nsubpartitions, :]

    # CDF looks similar for all values of nwait
    # nwait = 9
    # df = df[df.nwait .== nwait, :]

    # samples = Vector{Union{Float64,Missing}}()
    samples = Vector{Float64}()
    samples_fast = Vector{Float64}()
    samples_slow = Vector{Float64}()

    jobids = unique(df.jobid)
    println("Computing staleness over $(length(jobids)) jobs")
    for jobid in jobids
        M  = staleness_matrix_from_jobid(df, jobid, nworkers)
        append!(samples, skipmissing(vec(M)))

        # separate samples from the fastest half and slowest half of workers
        p = sortperm([mean(view(M, :, worker)) for worker in 1:nworkers])
        il = floor(Int, nworkers/2)
        iu = ceil(Int, nworkers/2)
        for i in 1:il
            append!(samples_fast, skipmissing(view(M, :, p[i])))
        end
        for i in iu:nworkers
            append!(samples_slow, skipmissing(view(M, :, p[i])))
        end
    end

    # all samples
    plt.figure()
    sort!(samples)
    cdf = (0:length(samples)-1) ./ (length(samples)-1)
    ccdf = 1 .- cdf
    plt.semilogy(samples, ccdf, label="All")

    # fastest half
    sort!(samples_fast)
    cdf = (0:length(samples_fast)-1) ./ (length(samples_fast)-1)
    ccdf = 1 .- cdf
    plt.semilogy(samples_fast, ccdf, label="Fastest 50%")

    # slowest half
    sort!(samples_slow)
    cdf = (0:length(samples_slow)-1) ./ (length(samples_slow)-1)
    ccdf = 1 .- cdf
    plt.semilogy(samples_slow, ccdf, label="Slowest 50%")

    plt.grid()
    plt.xlabel("Compute latency")
    plt.ylabel("CCDF")
    plt.legend()
    plt.ylim(1e-4, 1.0)
    # plt.xlim(0, 50)
    return
end