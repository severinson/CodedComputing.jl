# Code for analyzing and plotting convergence

"""

Return a timeseries composed of the best error for each time interval
"""
function get_best_timeseries(df, ts=range(minimum(df.time), maximum(df.time), length=50))
    xs = zeros(length(ts))
    ys = zeros(length(ts))
    for (i, t) in enumerate(ts)
        df_t = df[df.time .<= t, :]
        j = argmax(df_t.mse)
        xs[i] = df_t.time[j]
        ys[i] = df_t.mse[j]
        println("Best for t <= $(t): $(df_t.jobid[j]) (Np: $(df_t.nsubpartitions[j]), Nw: $(df_t.nwait[j]), η: $(df_t.stepsize[j])")
    end
    xs, ys
end

"""

Plot the best error over all experiments and parameter combinations
"""
function plot_timeseries_best(dct; opt=nothing)
    plt.figure()
    for (label, df) in dct
        println(label)
        df = df[ismissing.(df.mse) .== false, :]
        # df = df[df.nwait .< df.nworkers, :]
        if size(df, 1) == 0
            continue
        end
        xs, ys = get_best_timeseries(df)
        println()

        pf = plt.plot
        if !isnothing(opt)
            ys = opt .- ys
            pf = plt.semilogy
        end

        pf(xs, ys, "-o", label=label)
    end
    plt.xlabel("Time [s]")
    plt.ylabel("Explained variance")
    plt.grid()
    plt.legend()
    return
end

plot_timeseries_best(df::AbstractDataFrame; kwargs...) = plot_timeseries_best(Dict("df"=>df); kwargs...)

"""

Plot the MSE as a function of time (or iteration) separately for each unique job.
"""
function plot_timeseries(df; time=true, filters=Dict{String,Any}(), prune=false, opt=nothing)
    for (key, value) in filters
        df = df[df[key] .== value, :]
    end

    # pruning parameters
    m, atol, rtol = maximum(df.mse), 1e-6, 1

    plt.figure()
    for jobid in unique(df.jobid)
        df_jobid = df[df.jobid .== jobid, :]

        # optionally prune jobs that don't converge
        if prune && maximum(df_jobid.mse) < m*rtol - atol
            continue
        end

        ys = df_jobid.mse
        pf = plt.plot
        if !isnothing(opt)
            ys = opt .- ys
            pf = plt.semilogy
        end

        # plot convergence
        label = "job $jobid"
        if time            
            pf(df_jobid.time, ys, ".-", label=label)
        else
            pd(df_jobid.iteration, ys, ".-", label=label)
        end
    end
    if time
        plt.xlabel("Time [s]")
    else
        plt.xlabel("Iteration")
    end
    plt.ylabel("Explained variance")
    plt.legend()
    plt.grid()
    return
end



function plot_convergence(df, nworkers, opt=maximum(skipmissing(df.mse)); latency="empirical")
    # df = df[df.nworkers .== nworkers, :] 
    # df = df[df.nreplicas .== 1, :]
    # df = df[.!ismissing.(df.mse), :]

    df = filter(:nworkers => (x)->x==nworkers, df)
    df = filter(:nreplicas => (x)->x==1, df)
    df = filter(:mse => (x)->!ismissing(x), df)

    println("nworkers: $nworkers, opt: $opt")

    # (nwait, nsubpartitions, stepsize)
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
            (18, nsubpartitions, 0.9),        
            (27, nsubpartitions, 0.9),
        ]
    elseif nworkers == 72
        nsubpartitions = 160
        params = [
            (1, nsubpartitions, 0.9),            
            (3, nsubpartitions, 0.9),            
            (6, nsubpartitions, 0.9),                        
            (9, nsubpartitions, 0.9),
            # (18, nsubpartitions, 0.9),        
            # (27, nsubpartitions, 0.9),
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
    end

    plt.figure()    

    # # plot the bound
    # r = 2
    # Nw = 1
    # samp = 1

    # # get the convergence per iteration for batch GD
    # dfi = df
    # dfi = dfi[dfi.nsubpartitions .== 1, :]    
    # dfi = dfi[dfi.nwait .== nworkers, :]
    # dfi = dfi[dfi.stepsize .== 1, :]
    # dfi = dfi[dfi.variancereduced .== false, :]
    # dfi = dfi[dfi.kickstart .== false, :]
    # dfi = dfi[dfi.nostale .== false, :]
    # dfj = combine(groupby(dfi, :iteration), :mse => mean)
    # ys = opt .- dfj.mse_mean

    # # compute the iteration time for a scheme with a factor r replication
    # @assert length(unique(dfi.worker_flops)) == 1
    # worker_flops = r*unique(dfi.worker_flops)[1]
    # x0 = get_offset(worker_flops) .+ samp .* get_slope(worker_flops, nworkers) * Nw
    # xs = x0 .* (1:maximum(dfi.iteration))

    # # make the plot
    # plt.semilogy(xs, ys, "--k", label="Bound r: $r, Nw: $Nw")
    # write_table(xs, ys, "./data/bound_$(nworkers)_$(Nw)_$(r).csv")

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
        filename = "./dsag_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"
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
            write_table(xs, ys, filename)
        end

        # ### SAG
        # dfj = dfi
        # dfj = dfj[dfj.variancereduced .== true, :]
        # if nwait < nworkers # for nwait = nworkers, DSAG and SAG are the same
        #     dfj = dfj[dfj.nostale .== true, :]            
        # end        
        # filename = "./data/sag_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"
        # println("SAG: $(length(unique(dfj.jobid))) jobs")
        # if size(dfj, 1) > 0
        #     dfj = combine(groupby(dfj, :iteration), :mse => mean, :time => mean)
        #     if size(dfj, 1) > 0
        #         xs = dfj.time_mean
        #         ys = opt.-dfj.mse_mean                
        #         plt.semilogy(xs, ys, "^-", label="SAG (Nw: $nwait, Np: $nsubpartitions, η: $stepsize)")
        #         # write_table(xs, ys, filename)                
        #     end
        # end

        # ### SGD
        # dfj = dfi
        # dfj = dfj[dfj.variancereduced .== false, :]
        # filename = "./data/sgd_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"
        # println("SGD: $(length(unique(dfj.jobid))) jobs")
        # if size(dfj, 1) > 0
        #     dfj = combine(groupby(dfj, :iteration), :mse => mean, :time => mean)    
        #     if size(dfj, 1) > 0
        #         xs = dfj.time_mean
        #         ys = opt.-dfj.mse_mean                
        #         plt.semilogy(xs, ys, "s-", label="SGD (Nw: $nwait, Np: $nsubpartitions, η: $stepsize)")
        #         # write_table(xs, ys, filename)
        #     end
        # end
        
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
    println("SAG p: $nsubpartitions, $(length(unique(dfi.jobid))) jobs")
    dfj = by(dfi, :iteration, :mse => mean => :mse, :time => mean => :time)
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
        filename = "./sag_$(nworkers)_$(nsubpartitions)_$(stepsize).csv"
        write_table(xs, ys, filename)
    end
    # end

    # # Plot SGD
    # # nsubpartitions = 80
    # println("SGD p: $nsubpartitions")
    # stepsize = 0.9
    # dfi = df
    # dfi = dfi[dfi.nwait .== nworkers, :]
    # dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
    # dfi = dfi[dfi.variancereduced .== false, :]
    # dfi = dfi[dfi.stepsize .== stepsize, :]
    # dfj = by(dfi, :iteration, :mse => mean, :time => mean)
    # if size(dfj, 1) > 0
    #     xs = dfj.time_mean
    #     ys = opt.-dfj.mse_mean
    #     plt.semilogy(xs, ys, "c^-", label="SGD")
    # end        

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
        filename = "./gd_$(nworkers)_$(nsubpartitions)_$(stepsize).csv"
        write_table(xs, ys, filename)
    end    


    plt.xlim(1e-2, 1e2)
    plt.xscale("log")
    plt.grid()
    plt.legend()    
    plt.xlabel("Time [s]")
    plt.ylabel("Explained Variance Sub-optimality Gap")
    return
end