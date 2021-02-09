# Code for analyzing and plotting latency

# linear model
get_βs() = [0.005055059937837611, 8.075122937312302e-8, 1.1438758464435006e-16]
get_γs() = [0.03725188744901591, 3.109510011653974e-8, 6.399147477943208e-16]
get_offset(w) = 0.005055059937837611 .+ 8.075122937312302e-8w .+ 1.1438758464435006e-16w.^2
get_slope(w, nworkers) = 0.03725188744901591 .+ 3.109510011653974e-8(w./nworkers) .+ 6.399147477943208e-16(w./nworkers).^2

# shifted exponential model
get_shift(w) = 0.2514516116132241 .+ 6.687583396247953e-8w .+ 2.0095825408761404e-16w.^2
get_scale(w) = 0.23361469930191084 .+ 7.2464826067975726e-9w .+ 5.370433628859458e-17w^2

"""

Fit a shifted exponential latency model to the data.
"""
function fit_shiftexp_model(df, worker_flops)
    # df = df[df.nwait .== nwait, :]
    df = df[df.worker_flops .== worker_flops, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.kickstart .== false, :]    
    df = df[df.pfraction .== 1, :]

    # get the shift from waiting for 1 worker
    shift = quantile(df[df.nwait .== 1, :t_compute], 0.01)
    ts = df.t_compute .- shift

    # get the scale from waiting for all workers
    β = 0.0        
    for nworkers in unique(df.nworkers)
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        nwait = nworkers
        ts = dfi[dfi.nwait .== nwait, :t_compute] .- shift
        # σ = var(ts)
        # β1 = sqrt(σ / sum(1/i^2 for i in (nworkers-nwait+1):nworkers))        
        μ = mean(ts)
        βi = μ / sum(1/i for i in (nworkers-nwait+1):nworkers)
        β += βi * size(dfi, 1) / size(df, 1)
    end
    return shift, β
end

"""

Plot the shifted exponential shift and scale as a function of w.
"""
function plot_shiftexp_model(df)
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

    poly = Polynomials.fit(ws, shifts, 2)
    println(poly.coeffs)    
    ts = range(0, maximum(df.worker_flops), length=100)
    plt.plot(ts, poly.(ts))

    plt.grid()
    plt.xlabel("w")    
    plt.ylabel("shift")

    plt.figure()
    plt.plot(ws, scales, "o")

    poly = Polynomials.fit(ws, scales, 2)
    println(poly.coeffs)
    ts = range(0, maximum(df.worker_flops), length=100)
    plt.plot(ts, poly.(ts))    

    plt.grid()
    plt.xlabel("w")    
    plt.ylabel("scale")    

    return
end

"""

Return the number of workers that minimizes t_compute, when the workload is `σ`,
and the coordinator waits for the fastest `f` fraction of workers.
"""
function optimize_nworkers(σ0, f)
    βs = get_βs()
    γs = get_γs()
    c1 = γs[1]*f
    c2 = (βs[2] + γs[2]*f)*σ0
    sqrt(c2) / sqrt(c1)    
end

"""

Plot t_compute as a function of Nn for some value of σ0
σ0=1.393905852e9 is the workload associated with processing all data on 1 worker
"""
function plot_predictions(σ0=1.393905852e9; df=nothing)

    nworkers_all = 1:50
    σ0s = 10.0.^range(5, 12, length=20)    

    # plot the speedup due to waiting for fewer workers    
    for fi in [0.1, 0.5]
        f1 = fi
        f2 = f
        nws1 = optimize_nworkers.(σ0s, f1)
        nws2 = optimize_nworkers.(σ0s, f2)
        ts1 = get_offset.(σ0s./nws1) .+ get_slope.(σ0s./nws1, nws1) .* f1 .* nws1
        ts2 = get_offset.(σ0s./nws2) .+ get_slope.(σ0s./nws2, nws2) .* f2 .* nws2
        plt.semilogx(σ0s, ts2 ./ ts1, label="f: $fi")
    end
    plt.xlabel("σ0")
    plt.ylabel("speedup")
    plt.grid()
    plt.legend()          

    # plot the optimized t_compute as a function of σ0
    plt.figure()    
    for f in [0.1, 0.5, 1.0]
        nws = optimize_nworkers.(σ0s, f)
        ts = get_offset.(σ0s ./ nws) .+ get_slope.(σ0s./nws, nws) .* f .* nws
        plt.loglog(σ0s, ts, label="f: $f")

        # print values
        # println("f: $f")
        # for i in 1:length(nws)
        #     println("$(σ0s[i]) $(ts[i])")
        # end                
    end
    # plt.ylim(0, 10)
    plt.xlabel("σ0")
    plt.ylabel("T_compute*")
    plt.grid()
    plt.legend()
    # return
    
    # plot the optimized number of workers as a function of σ0
    plt.figure()
    for f in [0.1, 0.5, 1.0]
        nws = optimize_nworkers.(σ0s, f)
        plt.loglog(σ0s, nws, label="f: $f")

        # print values
        # println("f: $f")
        # for i in 1:length(nws)
        #     println("$(σ0s[i]) $(nws[i])")
        # end        
    end
    # plt.ylim(0, 10)
    plt.xlabel("σ0")
    plt.ylabel("Nn*")
    plt.grid()
    plt.legend()    

    # fix total amount of work    
    plt.figure()        
    for nsubpartitions in [1, 3, 20]
        f = 1.0
        npartitions = nworkers_all .* nsubpartitions
        ts = get_offset.(σ0 ./ npartitions) .+ get_slope.(σ0 ./ npartitions, nworkers_all) .* f .* nworkers_all
        plt.plot(nworkers_all, ts, label="Np: $nsubpartitions")

        # println("nsubpartitions: $nsubpartitions")
        # for i in 1:length(ts)
        #     println("$(nworkers_all[i]) $(ts[i])")
        # end

        # # point at which the derivative with respect to nworkers is zero
        σ = σ0/nsubpartitions
        x = optimize_nworkers(σ, f)

        if x <= length(ts)
            plt.plot([x], ts[round(Int, x)], "o")
            println("Np: $nsubpartitions, x: $x, t: $(ts[round(Int, x)])")        
        end

        f = 0.5
        npartitions = nworkers_all .* nsubpartitions
        ts = get_offset.(σ0 ./ npartitions) .+ get_slope.(σ0 ./ npartitions, nworkers_all) .* f .* nworkers_all
        plt.plot(nworkers_all, ts, "--", label="Np: $nsubpartitions (1/2)")

        println("nsubpartitions: $nsubpartitions")
        for i in 1:length(ts)
            println("$(nworkers_all[i]) $(ts[i])")
        end

        # # point at which the derivative with respect to nworkers is zero
        σ = σ0/nsubpartitions
        x = optimize_nworkers(σ, f)

        if x <= length(ts)
            plt.plot([x], ts[round(Int, x)], "o")
            println("Np: $nsubpartitions, x: $x, t: $(ts[round(Int, x)])")        
        end                
    end

    # Np = 3
    f = 1.0
    dfi = df
    dfi = dfi[dfi.kickstart .== false, :]
    dfi = dfi[dfi.nreplicas .== 1, :]
    dfi = dfi[dfi.pfraction .== 1, :]
    dfi = dfi[dfi.nsubpartitions .== 3, :]
    dfi = dfi[dfi.nwait .== round.(Int, f.*dfi.nworkers), :]
    dfj = combine(groupby(dfi, :nworkers), :t_compute => mean)
    plt.plot(dfj.nworkers, dfj.t_compute_mean, "s")
    plt.plot(dfi.nworkers, dfi.t_compute, ".")
    for i in 1:size(dfj, 1)
        println("$(dfj.nworkers[i]) $(dfj.t_compute_mean[i])")
    end
    println()
    println((minimum(dfi.worker_flops.*dfi.nworkers), maximum(dfi.worker_flops.*dfi.nworkers)))

    # expression for α1 + α2*(f*nworkers)
    # (to make sure it's correct)
    # ts = [βs[1] + γs[1]*f*nworkers + (βs[2]+γs[2]*f)*σ0/nworkers + βs[3]*(σ0/nworkers)^2 + γs[3]*f*σ0^2/nworkers^3 for nworkers in nworkers_all]
    # plt.plot(nworkers_all, ts, "--")

    plt.ylim(0, 10)
    plt.xlabel("Nn")
    plt.ylabel("T_compute [s]")
    plt.title("Fix total amount of work")    
    plt.grid()
    plt.legend()        
    return
end

"""

Plot the CCDF of the iteration time for all values of `nwait` for the given number of workers.
"""
function plot_iterationtime_cdf(df; nworkers::Integer=12)
    df = df[df.nworkers .== nworkers, :]
    plt.figure()
    for nwait in sort!(unique(df.nwait))
        df_nwait = df[df.nwait .== nwait, :]
        x = sort(df_nwait.t_compute)
        y = 1 .- range(0, 1, length=length(x))
        plt.semilogy(x, y, label="($nworkers, $nwait")
    end
    plt.ylim(1e-2, 1)
    plt.xlabel("Iteration time [s]")
    plt.ylabel("CCDF")
    plt.grid()
    plt.legend()
    plt.show()
end

"""

Plot the quantiles of the iteration time as a function of `nwait`, i.e., the number of 
workers waited for in each iteration.
"""
function plot_iterationtime_quantiles(dct)
    plt.figure()
    for (label, df) in dct
        df = df[df.nreplicas .== 1, :]

        df.nwait = df.nwait ./ df.nworkers

        offsets = Vector{Float64}()
        slopes = Vector{Float64}()
        flops = Vector{Float64}()
        for (nreplicas, nsubpartitions, worker_flops) in Iterators.product(unique(df.nreplicas), unique(df.nsubpartitions), unique(df.worker_flops))
            dfi = df
            dfi = dfi[dfi.nreplicas .== nreplicas, :]
            dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
            dfi = dfi[dfi.worker_flops .== worker_flops, :]
            # dfi = dfi[dfi.kickstart .!= true, :]
            if size(dfi, 1) == 0
                continue
            end

            xs = Vector{Float64}()
            ys = Vector{Float64}()
            mins = Vector{Float64}()
            maxes = Vector{Float64}()
            for nwait in unique(dfi.nwait)
                dfj = dfi[dfi.nwait .== nwait, :]
                if size(dfj, 1) == 0
                    continue
                end
                push!(xs, nwait)
                push!(ys, mean(dfj.t_compute))
                push!(mins, quantile(dfj.t_compute, 0.1))
                push!(maxes, quantile(dfj.t_compute, 0.9))
            end
            l = label * " nrep: $nreplicas, nsubp: $nsubpartitions, nflops: $(round(worker_flops, sigdigits=3))"            
            yerr = zeros(2, length(xs))
            yerr[1, :] .= ys .- mins
            yerr[2, :] .= maxes .- ys
            plt.errorbar(xs, ys, yerr=yerr, fmt=".", label=l)

            # plot a linear model fit to the data
            poly = Polynomials.fit(float.(dfi.nwait), float.(dfi.t_compute), 1)
            offset, slope = poly.coeffs
            t = range(0.0, maximum(xs), length=100)
            plt.plot(t, poly.(t))

            push!(offsets, offset)
            push!(slopes, slope)
            push!(flops, worker_flops)

            println("[nreplicas: $nreplicas, nsubp: $nsubpartitions, nflops: $worker_flops] offset: $(round(offset, digits=5)) ($(offset / worker_flops)), slope: $(round(slope, digits=5)) ($(slope / worker_flops))")
        end

        plt.grid()
        plt.legend()
        plt.xlabel("nwait")
        plt.ylabel("Compute time [s]")        

        # offset
        plt.figure()
        p = sortperm(flops)
        plt.plot(flops[p], offsets[p], "o")

        poly = Polynomials.fit(flops, offsets, 2)
        t = range(0.0, maximum(df.worker_flops), length=100)
        plt.plot(t, poly.(t))

        plt.grid()
        plt.legend()
        plt.xlabel("flops")
        plt.ylabel("offset")     

        # # print the parameters
        # for i in 1:length(flops)        
        #     println("$(flops[i]) $(offsets[i])")
        # end

        # # print the fitted line
        # println()
        # for i in 1:length(t)        
        #     println("$(t[i]) $(poly(t[i]))")
        # end        

        # print the quadratic parameters
        p = poly.coeffs
        println("offset")
        println("$(p[1]) & $(p[2]) & $(p[3])")
        
        # slope
        plt.figure()
        p = sortperm(flops)
        plt.plot(flops[p], slopes[p], "o")

        poly = Polynomials.fit(flops, slopes, 2)
        t = range(0.0, maximum(flops), length=100)
        plt.plot(t, poly.(t))          

        # # print the parameters
        # for i in 1:length(flops)        
        #     println("$(flops[i]) $(slopes[i])")
        # end

        # # print the fitted line
        # println()
        # for i in 1:length(t)        
        #     println("$(t[i]) $(poly(t[i]))")
        # end                

        # print the quadratic parameters
        println("slope")
        p = poly.coeffs
        println("$(p[1]) & $(p[2]) & $(p[3]) \\")        

        plt.grid()
        plt.legend()
        plt.xlabel("flops")
        plt.ylabel("slope")             
    end



    # tikzplotlib.save("./plots/tcompute.tex")
    
    return
end

plot_iterationtime_quantiles(df::AbstractDataFrame) = plot_iterationtime_quantiles(Dict("df"=>df))

"""

Return a DataFrame of linear model parameters for t_compute fit to the data
"""
function linear_model_df(df)
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    rv = DataFrame()    
    for (nworkers, worker_flops) in Iterators.product(unique(df.nworkers), unique(df.worker_flops))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        if size(dfi, 1) < 2
            continue
        end
        poly = Polynomials.fit(float.(dfi.nwait), float.(dfi.t_compute), 1)
        # println(poly)
        offset, slope = poly.coeffs
        row = Dict("nworkers" => nworkers, "worker_flops" => worker_flops, "offset" => offset, "slope" => slope)
        push!(rv, row, cols=:union)
    end
    sort!(rv, [:nworkers, :worker_flops])
    rv
end

"""

Plot the linear model parameters 
"""
function plot_compute_time_model(df)
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.pfraction .== 1, :]

    dfm = linear_model_df(df)

    # offset
    plt.figure()
    for nworkers in unique(dfm.nworkers)
        dfi = dfm
        dfi = dfi[dfi.nworkers .== nworkers, :]        
        plt.semilogx(dfi.worker_flops, dfi.offset, "o", label="Nn: $nworkers")        

        # print parameters
        # println("Nn: $nworkers")
        # sort!(dfi, [:worker_flops])
        # for i in 1:size(dfi, 1)
        #     println("$(dfi.worker_flops[i]) $(dfi.offset[i])")
        # end
    end    

    # quadratic fit
    poly, coeffs = fit_polynomial(float.(dfm.worker_flops), float.(dfm.offset), 2)    
    t = range(0, maximum(dfm.worker_flops), length=100)
    plt.semilogx(t, poly.(t))

    # print fit line
    println(coeffs)
    # for i in 1:length(t)
    #     println("$(t[i]) $(poly(t[i]))")
    # end

    plt.grid()
    plt.xlabel("flops")
    plt.ylabel("offset")
    plt.legend()

    # slope
    plt.figure()
    for nworkers in unique(dfm.nworkers)
        dfi = dfm
        dfi = dfi[dfi.nworkers .== nworkers, :]
        x = dfi.worker_flops ./ dfi.nworkers # mysterious normalization
        plt.semilogx(x, dfi.slope, "o", label="Nn: $nworkers")

        # print parameters
        # println("Nn: $nworkers")
        # sort!(dfi, [:worker_flops])
        # for i in 1:size(dfi, 1)
        #     println("$(x[i]) $(dfi.slope[i])")
        # end        
    end

    # quadratic fit
    poly, coeffs = fit_polynomial(float.(dfm.worker_flops ./ dfm.nworkers), float.(dfm.slope), 2)
    t = range(0, maximum(dfm.worker_flops ./ dfm.nworkers), length=100)
    plt.semilogx(t, poly.(t))

    # print fit line
    println(coeffs)
    # for i in 1:length(t)
    #     println("$(t[i]) $(poly(t[i]))")
    # end    

    plt.grid()
    plt.xlabel("flops / Nn")
    plt.ylabel("slope")
    plt.legend()
    return
end

"""

Plots:
- Empirical latency (samples and sample average)
- Latency predicted by the proposed linear model
- Latency predicted by the i.i.d. shifted exponential model

"""
function plot_compute_time(df, nworkers=18)
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.pfraction .== 1, :]    
    df = df[df.nworkers .== nworkers, :]

    for nsubpartitions in sort!(unique(df.nsubpartitions))
        dfi = df
        dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
        @assert length(unique(dfi.worker_flops)) == 1    
        w = unique(dfi.worker_flops)[1]    

        # scatter plot of the samples
        plt.plot(dfi.nwait, dfi.t_compute, ".")    
        write_table(dfi.nwait, dfi.t_compute, "./data/model_raw.csv")

        # compute average delay and quantiles
        # and plot it
        dfj = combine(
            groupby(dfi, :nwait), 
            :t_compute => mean, 
            :t_compute => ((x)->quantile(x, 0.1)) => :q1,
            :t_compute => ((x)->quantile(x, 0.9)) => :q9,
            )
        sort!(dfj, :nwait)
        plt.plot(dfj.nwait, dfj.t_compute_mean, "o")
        write_table(dfj.nwait, dfj.t_compute_mean, "./data/model_means.csv")    

        println("Latency average p: $nworkers * $nsubpartitions, w: $w")
        for i in 1:size(dfj, 1)
            println("$(dfj[i, :nwait]) $(dfj[i, :t_compute_mean])")
        end

        # plot predicted delay (by a local model)
        poly = fit_polynomial(float.(dfi.nwait), float.(dfi.t_compute), 1)
        xs = [0, nworkers]
        plt.plot(xs, poly.(xs), "--", label="Local model")            
        println(poly)

        # # print values
        # println("local model")
        # for i in 1:length(xs)
        #     println("$(xs[i]) $(ys[i])")
        # end

        # plot predicted delay (by the global model)
        println("w: $w")
        xs = 1:nworkers
        ys = get_offset(w) .+ get_slope(w, nworkers) .* xs
        plt.plot(xs, ys)
        # write_table(xs, ys, "./data/model_linear.csv")

        # # plot delay predicted by the shifted exponential order statistics model
        # # shift, β = fit_shiftexp_model(df, w)
        # shift = get_shift(w)
        # scale = get_scale(w)
        # ys = [mean(ExponentialOrder(scale, nworkers, nwait)) for nwait in xs] .+ shift
        # plt.plot(xs, ys, "--")
        # # write_table(xs, ys, "./data/model_shiftexp.csv")
    end

    plt.grid()
    plt.xlabel("Nw")
    plt.ylabel("T_compute")
    return
end

"""

Plot the update time at the master against the number of sub-partitions.
"""
function plot_update_time(dct)

    # SAG
    plt.figure()    
    df = dct["pcavr"]
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.nostale .== false, :]
    npartitions = df.nworkers .* df.nsubpartitions
    # df.cost = min.(npartitions, 2df.nwait)
    df.cost = df.nsubpartitions
    # df.cost = df.nwait .+ min.(npartitions, 2df.nwait)
    for nworkers in sort!(unique(df.nworkers))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        # plt.plot(dfi.cost, dfi.t_update, "o", label="Nn: $nworkers")

        dfj = combine(groupby(dfi, :cost), :t_update => mean)
        sort!(dfj, :cost)
        plt.plot(dfj.cost, dfj.t_update_mean, "o", label="Nn: $nworkers")

        # print values
        # println("Nn=$nworkers")
        # for i in 1:size(dfj, 1)
        #     println("$(dfj.cost[i]) $(dfj.t_update_mean[i])")
        # end
    end
    plt.legend()
    plt.grid()
    plt.xlabel("p / Nn")
    plt.ylabel("Update time [s]")

    # SGD
    plt.figure()    
    df = dct["pca"]
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    npartitions = df.nworkers .* df.nsubpartitions
    # df.cost = min.(npartitions, 2df.nwait)
    # df.cost = npartitions
    df.cost = df.nwait
    for nworkers in sort!(unique(df.nworkers))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        # plt.plot(dfi.cost, dfi.t_update, "o", label="Nn: $nworkers")

        dfj = combine(groupby(dfi, :cost), :t_update => mean)
        sort!(dfj, :cost)
        plt.plot(dfj.cost, dfj.t_update_mean, "o", label="Nn: $nworkers")

        # print values
        println("Nn=$nworkers")
        for i in 1:size(dfj, 1)
            println("$(dfj.cost[i]) $(dfj.t_update_mean[i])")
        end        
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Nw")
    plt.ylabel("Update time [s]")

    return 

    for (label, df) in dct
        df = df[df.nreplicas .== 1, :]

        for (nreplicas, nsubpartitions) in Iterators.product(unique(df.nreplicas), unique(df.nsubpartitions))

        

            dfi = df
            dfi = dfi[dfi.nreplicas .== nreplicas, :]
            dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
            dfi = dfi[Missings.replace(dfi.kickstart, false) .== false, :]
            dfi = dfi[dfi.iteration .> 1, :] # ignore the first iteration
            # dfi = dfi[dfi.iteration .== 1, :] # only first iteration
            if size(dfi, 1) == 0                
                continue
            end

            xs = Vector{Float64}()
            ys = Vector{Float64}()
            mins = Vector{Float64}()
            maxes = Vector{Float64}()
            for nwait in unique(dfi.nwait)
                dfj = dfi[dfi.nwait .== nwait, :]
                if size(dfj, 1) == 0
                    continue
                end
                push!(xs, nwait)
                push!(ys, mean(dfj.t_update))
                push!(mins, quantile(dfj.t_update, 0.1))
                push!(maxes, quantile(dfj.t_update, 0.9))

                k = argmax(dfj.t_update)
                println((nreplicas, nsubpartitions, nwait, dfj.jobid[k]))

                # plt.plot(dfj.nwait, dfj.t_update, ".")
            end
            l = label * " nreplicas: $nreplicas, nsubpartitions: $nsubpartitions"
            yerr = zeros(2, length(xs))
            yerr[1, :] .= ys .- mins
            yerr[2, :] .= maxes .- ys
            # plt.errorbar(xs, ys, yerr=yerr, fmt=".", label=l)
            plt.plot(xs, ys, ".", label=l)

            # # plot a linear model fit to the data
            # offset, slope = linear_model(xs, ys)
            # plt.plot([0, maximum(xs)], offset .+ [0, maximum(xs)*slope])
            # println("[nreplicas: $nreplicas, nsubpartitions: $nsubpartitions] offset: $offset ($(offset / nsubpartitions)), slope: $slope ($(slope / nsubpartitions))")
        end
    end

    plt.grid()
    plt.legend()
    plt.xlabel("nwait")
    plt.ylabel("Update time [s]")

    # tikzplotlib.save("./plots/tupdate.tex")
    
    return  
end

plot_update_time(df::AbstractDataFrame) = plot_update_time(Dict("df"=>df))

function plot_compute_time_3d(df)

    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]

    colors = Iterators.cycle(["r", "b", "g", "k", "m"])

    # fix nworkers
    fig = plt.figure()
    ax = fig[:add_subplot](111, projection="3d")    
    for nworkers in unique(df.nworkers)
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        if size(dfi, 1) == 0
            continue
        end
        ax[:plot](dfi.nwait, dfi.worker_flops, dfi.t_compute, ".", label="Nn: $nworkers")
    end
    plt.xlabel("nwait")
    plt.ylabel("nflops")
    ax[:set_zlabel]("Compute time [s]")
    plt.grid()
    plt.legend()    

    # fix nwait
    fig = plt.figure()
    ax = fig[:add_subplot](111, projection="3d")        
    for nwait in unique(df.nwait)
        dfi = df
        dfi = dfi[dfi.nwait .== nwait, :]
        if size(dfi, 1) == 0
            continue
        end
        ax[:plot](dfi.nworkers, dfi.worker_flops, dfi.t_compute, ".", label="Nw: $nwait")
    end    
    plt.xlabel("nworkers")
    plt.ylabel("nflops")
    ax[:set_zlabel]("Compute time [s]")
    plt.grid()
    plt.legend()

    # fix worker_flops
    fig = plt.figure()
    ax = fig[:add_subplot](111, projection="3d")        
    for nflops in unique(df.worker_flops)
        dfi = df
        dfi = dfi[dfi.worker_flops .== nflops, :]
        if size(dfi, 1) == 0
            continue
        end
        ax[:plot](dfi.nworkers, dfi.nwait, dfi.t_compute, ".", label="flops: $nflops")
    end    
    plt.xlabel("nworkers")
    plt.ylabel("nwait")
    ax[:set_zlabel]("Compute time [s]")
    plt.grid()
    plt.legend()    


    return
end

"""

Plot the compute time per iteration against the iteration index.
"""
function plot_compute_time_traces(df)
    plt.figure()
    for (nreplicas, nworkers, nwait, worker_flops) in Iterators.product(unique(df.nreplicas), unique(df.nworkers), unique(df.nwait), unique(df.worker_flops))
        dfi = df
        dfi = dfi[dfi.nreplicas .== nreplicas, :]
        dfi = dfi[dfi.nworkers .== nworkers, :]
        dfi = dfi[dfi.nwait .== nwait, :]
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        if size(dfi, 1) == 0
            continue
        end
        label = "$((nreplicas, nworkers, nwait, worker_flops))"
        plt.plot(dfi.iteration, dfi.t_compute, ".", label=label)
    end

    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Compute time [s]")
end

# Let's start considering the latency distribution of individual workers
# I realize now that I need a relatively large number of iterations per experiment to do this
# Since it's going to be different across experiments
# I'm also interested in seeing how much of the straggling is due to workload imbalance

# Let's continue investigating this today
# I also need to redo the plots from the paper using the new shuffled matrix to understand what needs to be fixed

function plot_individual_cdf(df, nworkers=18, nsubpartitions=5)
    df = df[df.nworkers .== nworkers, :]
    df = df[df.nsubpartitions .== nsubpartitions, :]
    df = df[df.nwait .== nworkers, :]
    for jobid in unique(df.jobid)
        dfi = df
        dfi = dfi[dfi.jobid .== jobid, :]
        if ismissing(dfi.latency_worker_1[1])
            continue
        end
        plt.figure()
        for i in 1:nworkers
            xs = sort!(dfi["latency_worker_$i"])
            ys = range(0, 1, length=length(xs))
            plt.plot(xs, ys, "-k")
        end
        return
    end
end