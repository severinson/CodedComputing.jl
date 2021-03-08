# Code for analyzing and plotting convergence

"""

Return a timeseries composed of the best error for each time interval
"""
function get_best_timeseries(df, ts=range(minimum(df.t_total), maximum(df.t_total), length=50))
    xs = zeros(length(ts))
    ys = zeros(length(ts))
    for (i, t) in enumerate(ts)
        df_t = df[df.t_total .<= t, :]
        j = argmax(df_t.mse)
        xs[i] = df_t.t_total[j]
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
            pf(df_jobid.t_total, ys, ".-", label=label)
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



function plot_genome_convergence(df, nworkers=unique(df.nworkers)[1], opt=maximum(skipmissing(df.mse)))
    println("nworkers: $nworkers, opt: $opt")

    # (nwait, nsubpartitions, stepsize)
    if nworkers == 6
        Np = 5
        params = [
            (nworkers, 1, 1.0), # full GD
            (1, Np, 0.9),
            (2, Np, 0.9),
            (3, Np, 0.9),
            (4, Np, 0.9),
            (5, Np, 0.9),
            (6, Np, 0.9),
        ]        
    elseif nworkers == 12

        # sub-partitioning plot
        # Nw = 12
        # params = [
        #     (Nw, 1, 1.0), # full GD
        #     (Nw, 2, 0.9),
        #     (Nw, 3, 0.9),
        #     (Nw, 4, 0.9),
        # ]

        Np = 5
        params = [
            (nworkers, 1, 1.0), # full GD
            (1, Np, 0.9),
            (3, Np, 0.9),
            (6, Np, 0.9),
            (10, Np, 0.9),
            (12, Np, 0.9),
        ]        

        # nwait plot
    elseif nworkers == 18

        # sub-partitioning plot
        # Nw = 18
        # params = [
        #     (Nw, 1, 1.0), # full GD
        #     (Nw, 2, 0.9),
        #     (Nw, 3, 0.9),
        #     (Nw, 4, 0.9),
        # ]

        # nwait plot
        Np = 2
        params = [
            (nworkers, 1, 1.0), # full GD
            # (9, Np, 0.6),
            # (9, Np, 0.7),
            # (9, Np, 0.8),
            # (9, Np, 0.9),
            # (9, Np, 0.8),
            (1, Np, 0.8),   
            (3, Np, 0.8),            
            (9, Np, 0.9),            
            (18, Np, 0.9),
        ]              
    elseif nworkers == 24
        params = [
            (nworkers, 1, 1), 
            (1, 1, 1),
            (nworkers, 3, 0.9),
            (1, 3, 0.9),
        ]    
    elseif nworkers == 36

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
            # (1, nsubpartitions, 0.9),            
            (3, nsubpartitions, 0.9),            
            # (6, nsubpartitions, 0.9),                        
            # (9, nsubpartitions, 0.9),            
            # (18, nsubpartitions, 0.9),        
            # (27, nsubpartitions, 0.9),
        ]
    end

    df = df[df.nworkers .== nworkers, :]    
    df = df[df.kickstart .== false, :]
    # df = df[df.nostale .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[.!ismissing.(df.mse), :]

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
        filename = "./data/dsag_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"
        println("DSAG: $(length(unique(dfj.jobid))) jobs")
        if size(dfj, 1) > 0
            dfj = combine(groupby(dfj, :iteration), :mse => mean => :mse, :t_total => mean => :t_total)
            if size(dfj, 1) > 0
                xs = dfj.t_total
                ys = opt.-dfj.mse
                plt.semilogy(xs, ys, ".-", label="DSAG w=$nwait, p=$nsubpartitions")
                # write_table(xs, ys, filename)
            end
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
        #     dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)
        #     if size(dfj, 1) > 0
        #         xs = dfj.t_total_mean
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
        #     dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)    
        #     if size(dfj, 1) > 0
        #         xs = dfj.t_total_mean
        #         ys = opt.-dfj.mse_mean                
        #         plt.semilogy(xs, ys, "s-", label="SGD (Nw: $nwait, Np: $nsubpartitions, η: $stepsize)")
        #         # write_table(xs, ys, filename)
        #     end
        # end
        
        println()
    end

    # Plot SAG
    # for nsubpartitions in sort!(unique(df.nsubpartitions))
    nsubpartitions = 80
    stepsize = 0.9
    dfi = df
    dfi = dfi[dfi.nwait .== nworkers, :]
    dfi = dfi[dfi.variancereduced .== true, :]
    dfi = dfi[dfi.stepsize .== stepsize, :]    
    dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
    println("SAG p: $nsubpartitions, $(length(unique(dfi.jobid))) jobs")
    dfj = by(dfi, :iteration, :mse => mean => :mse, :t_total => mean => :t_total)
    if size(dfj, 1) > 0
        xs = dfj.t_total
        ys = opt.-dfj.mse
        plt.semilogy(xs, ys, "o-", label="SAG p=$nsubpartitions")
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
    # dfj = by(dfi, :iteration, :mse => mean, :t_total => mean)
    # if size(dfj, 1) > 0
    #     xs = dfj.t_total_mean
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
    dfj = by(dfi, :iteration, :mse => mean => :mse, :t_total => mean => :t_total)
    if size(dfj, 1) > 0
        xs = dfj.t_total
        ys = opt.-dfj.mse
        plt.semilogy(xs, ys, "ms-", label="GD")
    end    


    plt.xlim(0)
    plt.grid()
    plt.legend()    
    plt.xlabel("Time [s]")
    plt.ylabel("Explained Variance Sub-optimality Gap")
    return
end