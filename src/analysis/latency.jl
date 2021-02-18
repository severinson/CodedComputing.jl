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
    # df = df[df.kickstart .== false, :]
    # df = df[df.nreplicas .== 1, :]
    rv = DataFrame()    
    for (nworkers, worker_flops) in Iterators.product(unique(df.nworkers), unique(df.worker_flops))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        if size(dfi, 1) < 2
            continue
        end

        # Polynomials.jl fit
        # poly = Polynomials.fit(float.(dfi.nwait), float.(dfi.t_compute), 1)        
        # offset, slope = poly.coeffs

        # homemade polynomial fit
        poly, coeffs = fit_polynomial(float.(dfi.nwait), float.(dfi.latency), 1)
        offset, slope = coeffs

        row = Dict("nworkers" => nworkers, "worker_flops" => worker_flops, "intercept" => offset, "slope" => slope)
        push!(rv, row, cols=:union)
    end
    sort!(rv, [:nworkers, :worker_flops])
    rv
end

"""

Return a DataFrame composed of the mean latency for each job.
"""
function mean_latency_df(df)
    df = by(df, :jobid, :latency => mean => :latency, :nworkers => mean => :nworkers, :nwait => mean => :nwait, :worker_flops => mean => :worker_flops)
    df.worker_flops .= round.(df.worker_flops, digits=6) # avoid rounding errors
    df
end

"""

Same as linear_model_df, except that the linear model parameters are fit to the average latency for each experiment.
"""
function mean_linear_model_df(df)
    # df = df[df.kickstart .== false, :]
    # df = df[df.nreplicas .== 1, :]
    df = mean_latency_df(df)
    df = by(df, [:nworkers, :worker_flops], [:nwait, :latency] => (x) -> NamedTuple{(:intercept, :slope)}(fit_polynomial(x.nwait, x.latency, 1)[2]))
    sort!(df, [:nworkers, :worker_flops])
end

"""

Plot the linear model parameters 
"""
function plot_linear_model(df)
    # df = df[df.kickstart .== false, :]
    # df = df[df.nreplicas .== 1, :]
    # df = df[df.pfraction .== 1, :]

    # dfm = linear_model_df(df)
    dfm = mean_linear_model_df(df)
    # dfm = order_linear_model_df(df)

    # offset
    plt.figure()
    for nworkers in unique(dfm.nworkers)
        dfi = dfm
        dfi = dfi[dfi.nworkers .== nworkers, :]        
        plt.semilogx(dfi.worker_flops, dfi.intercept, "o", label="Nn: $nworkers")        

        # print parameters
        # println("Nn: $nworkers")
        # sort!(dfi, [:worker_flops])
        # for i in 1:size(dfi, 1)
        #     println("$(dfi.worker_flops[i]) $(dfi.intercept[i])")
        # end
    end    

    # quadratic fit
    poly, coeffs = fit_polynomial(float.(dfm.worker_flops), float.(dfm.intercept), 2)    
    t = range(0, maximum(dfm.worker_flops), length=100)
    plt.semilogx(t, poly.(t))

    # print fit line
    # println(coeffs)
    # for i in 1:length(t)
    #     println("$(t[i]) $(poly(t[i]))")
    # end

    plt.grid()
    plt.xlabel("flops")
    plt.ylabel("offset")
    plt.legend()

    # intercept
    plt.figure()
    for nworkers in unique(dfm.nworkers)
        dfi = dfm
        dfi = dfi[dfi.nworkers .== nworkers, :]
        x = dfi.worker_flops ./ dfi.nworkers
        # x = dfi.worker_flops
        plt.semilogx(x, dfi.slope, "o", label="Nn: $nworkers")

        # print parameters
        println("Nn: $nworkers")
        sort!(dfi, [:worker_flops])
        for i in 1:size(dfi, 1)
            println("$(x[i]) $(dfi.slope[i])")
        end        
    end

    # quadratic fit
    poly, coeffs = fit_polynomial(float.(dfm.worker_flops ./ dfm.nworkers), float.(dfm.slope), 2)
    t = range(0, maximum(dfm.worker_flops ./ dfm.nworkers), length=100)
    # t = range(0, maximum(dfm.worker_flops), length=100)
    plt.semilogx(t, poly.(t))

    # print fit line
    # println(coeffs)
    for i in 1:length(t)
        println("$(t[i]) $(poly(t[i]))")
    end    

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
function plot_latency(df, nworkers=6)
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.pfraction .== 1, :]    
    df = df[df.nworkers .== nworkers, :]
    # df = mean_latency_df(df)

    plt.figure()
    for worker_flops in sort!(unique(df.worker_flops), rev=true)
        # for nsubpartitions in sort!(unique(df.nsubpartitions))
        dfi = df
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        # @assert length(unique(dfi.worker_flops)) == 1    
        # w = unique(dfi.worker_flops)[1]    
        println(mean(dfi.nsubpartitions))

        # scatter plot of the samples
        plt.plot(dfi.nwait, dfi.t_compute, ".", label="c: $(round(worker_flops, sigdigits=3))")    
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

        println("Latency average flops: $worker_flops")
        for i in 1:size(dfj, 1)
            println("$(dfj[i, :nwait]) $(dfj[i, :t_compute_mean])")
        end

        # plot predicted delay (by a local model)
        poly, coeffs = fit_polynomial(float.(dfi.nwait), float.(dfi.t_compute), 1)
        xs = [0, nworkers]
        ys = poly.(xs)
        plt.plot(xs, ys, "--")            
        println(poly)

        # print values
        println("local model")
        for i in 1:length(xs)
            println("$(xs[i]) $(ys[i])")
        end

        # plot predicted delay (by the global model)
        println("c: $worker_flops")
        xs = [0, nworkers]
        ys = get_offset(worker_flops) .+ get_slope(worker_flops, nworkers) .* xs
        plt.plot(xs, ys)
        # write_table(xs, ys, "./data/model_linear.csv")
        println("global model")
        for i in 1:length(xs)
            println("$(xs[i]) $(ys[i])")
        end        


        # # plot delay predicted by the shifted exponential order statistics model
        # # shift, β = fit_shiftexp_model(df, w)
        # shift = get_shift(w)
        # scale = get_scale(w)
        # ys = [mean(ExponentialOrder(scale, nworkers, nwait)) for nwait in xs] .+ shift
        # plt.plot(xs, ys, "--")
        # # write_table(xs, ys, "./data/model_shiftexp.csv")
    end
    plt.legend()
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

"""

Scatterplot of worker latency vs. density of the partitions stored by each worker.
"""
function plot_latency_density(df)
    nworkers = 18
    nsubpartitions = 1
    df = df[df.nworkers .== 18, :]
    df = df[df.nsubpartitions .== 1, :]
    df = orderstats_df(df)
    density = [0.05140742522974717, 0.05067826288956093, 0.05122862096280494, 0.050645562535343865, 0.05099657254820253, 0.05138775739534186, 0.050901895214905575, 0.050316781109837165, 0.05188810059429005, 0.051392694196391135, 0.05082197380306366, 0.050276556499358506, 0.05169503754425293, 0.051862277178989835, 0.05151591182965395, 0.05201310404786811, 0.05083756996715019, 0.05134404845100367]    
    plt.plot(density[df.worker_index], df.latency, ".")
    plt.grid()
    return
end

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
function plot_latency_traces(df; nworkers=nothing, nwait=nothing, nsubpartitions=nothing, jobid=nothing)
    plt.figure()
    if !isnothing(nworkers)    
        df = df[df.nworkers .== nworkers, :]
    end
    if !isnothing(nwait)    
        df = df[df.nwait .== nwait, :]
    end
    if !isnothing(nsubpartitions)    
        df = df[df.nsubpartitions .== nsubpartitions, :]
    end
    if !isnothing(jobid)    
        df = df[df.jobid .== jobid, :]
    end
    for jobid in sort(unique(df.jobid))
        dfi = df
        dfi = dfi[dfi.jobid .== jobid, :]        
        plt.plot(dfi.t_total, dfi.t_compute, ".", label="ID: $jobid")
    end
    plt.grid()
    plt.legend()
    plt.xlabel("Cumulative compute time [s]")
    plt.ylabel("Compute time [s]")
end

"""

Return a df composed of the order statistic samples for each worker, iteration, and job.
"""
function orderstats_df(df)    
    df = df[.!ismissing.(df["latency_worker_1"]), :]    
    if size(df, 1) == 0
        return DataFrame()
    end    
    nworkers = maximum(df.nworkers)

    # stack by worker latency
    df1 = stack(df, ["latency_worker_$i" for i in 1:nworkers], value_name=:worker_latency)
    df1[:worker_index] = parse.(Int, last.(split.(df1.variable, "_")))
    df1 = select(df1, Not(:variable))
    df1 = select(df1, Not(["repoch_worker_$i" for i in 1:nworkers]))
    if "compute_latency_worker_1" in names(df)
        df1 = select(df1, Not(["compute_latency_worker_$i" for i in 1:nworkers]))    
    end
    df1 = df1[.!ismissing.(df1.worker_latency), :]
    
    # stack by worker receive epoch
    df2 = stack(df, ["repoch_worker_$i" for i in 1:nworkers], [:jobid, :iteration], value_name=:repoch)
    df2[:worker_index] = parse.(Int, last.(split.(df2.variable, "_")))
    df2 = select(df2, Not(:variable))
    df2 = df2[.!ismissing.(df2.repoch), :]
    df2[:isstraggler] = df2.repoch .< df2.iteration
    joined = innerjoin(df1, df2, on=[:jobid, :iteration, :worker_index]) 

    # stack by worker compute latency
    if "compute_latency_worker_1" in names(df)
        df3 = stack(df, ["compute_latency_worker_$i" for i in 1:nworkers], [:jobid, :iteration], value_name=:worker_compute_latency)
        df3[:worker_index] = parse.(Int, last.(split.(df3.variable, "_")))
        df3 = select(df3, Not(:variable))
        df3 = df3[.!ismissing.(df3.worker_compute_latency), :]
        joined = innerjoin(joined, df3, on=[:jobid, :iteration, :worker_index])
    end

    # the latency of stragglers is infinite
    joined[joined.isstraggler, :worker_latency] .= Inf

    # compute the order of all workers
    sort!(joined, [:jobid, :iteration, :worker_latency])
    joined[:order] = by(joined, [:jobid, :iteration], :nworkers => ((x) -> collect(1:maximum(x))) => :order).order
    if "compute_latency_worker_1" in names(df)
        sort!(joined, [:jobid, :iteration, :worker_compute_latency])
        joined[:compute_order] = by(joined, [:jobid, :iteration], :nworkers => ((x) -> collect(1:maximum(x))) => :order).order        
    end

    return joined
end

"""

DataFrame composed of linear model parameters fit the order statistics latency.
"""
function order_model_df(df)
    df = df[df.nwait .== df.nworkers, :]
    df = orderstats_df(df)
    df = df[df.isstraggler .== false, :]    
    df = by(df, [:nworkers, :worker_flops], [:order, :worker_latency] => (x) -> NamedTuple{(:intercept, :x1, :x2, :x3)}(fit_polynomial(x.order, x.latency, 3)[end]))
    sort!(df, [:nworkers, :worker_flops])
end

function plot_order_model(df)
    df = order_model_df(df)
    
    for nworkers in sort!(unique(df.nworkers))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        plt.plot(dfi.worker_flops, dfi.intercept, ".", label="Nn: $nworkers")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("flops")
    plt.ylabel("Intercept")
    return
end

"""

Plot the latency of the w-th fastest worker as a function of worker_flops for all nworkers.
"""
function plot_order_latency(df, order=1)
    df = df[df.nwait .== df.nworkers, :]
    df = orderstats_df(df)
    df = df[df.isstraggler .== false, :]
    # df = df[df.order .== df.nworkers, :] # TODO: fix, not order 
    for nworkers in sort!(unique(df.nworkers))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        # plt.plot(dfi.worker_flops, dfi.latency, ".", label="Nn: $nworkers")
        dfj = by(dfi, :worker_flops, :worker_latency => mean)
        plt.plot(dfj.worker_flops, dfj.latency_mean ./ nworkers, "o", label="Nn: $nworkers")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("flops")
    plt.ylabel("Latency [s]")
    return
end



"""

Plot
- Latency order stats recorded individually for each worker for different w_target
- Iteration latency for different w_target
"""
function plot_orderstats(df, nworkers=18, work=1; onlycompute=true)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :compute_latency : :latency    
    df = df[df.nworkers .== nworkers, :]

    # filter by either number of partitions or worker_flops
    if typeof(work) <: Integer
        df = df[df.nsubpartitions .== work, :]
    elseif typeof(work) <: Real
        df = df[isapprox.(df.worker_flops, work, rtol=0.1), :]
    else
        error()
    end    
    println("worker_flops:\t$(unique(df.worker_flops))")
    println("nbytes:\t$(unique(df.nbytes))\n")
    # return df.t_update

    dfo = orderstats_df(df)    
    dfo.order = dfo.order ./ dfo.nworkers
    dfo.compute_order = dfo.compute_order ./ dfo.nworkers
    # dfo = dfo[dfo.order .<= dfo.nwait, :]

    # latency order stats recorded for individual workers
    plt.figure()
    for nwait_target in sort!(unique(dfo.nwait))
        # for f in [3/nworkers, 0.5, 1.0]
        # nwait_target = round(Int, f*nworkers)        
        
        dfi = dfo
        dfi = dfi[dfi.nwait .== nwait_target, :]
        # dfi = dfi[dfi.order .<= dfi.nwait, :]
        if size(dfi, 1) == 0
            continue
        end            
        # plt.plot(dfi.order, dfi.worker_latency, ".", label="w_target: $nwait_target")
        # straggler_fraction = sum(dfi.isstraggler) / length(dfi.isstraggler)
        straggler_fraction = sum(dfi[col] .< dfi.nwait) / size(dfi, 1)
        dfi = dfi[dfi[col] .* dfi.nworkers .<= dfi.nwait, :]
        # dfi = dfi[dfi.isstraggler .== false, :]

        dfj = by(dfi, col, :worker_compute_latency => mean => :mean, :jobid => ((x) -> length(unique(x))) => :njobs)
        sort!(dfj, col)
        plt.plot(dfj[col], dfj.mean, "o", label="w_target: $nwait_target")

        println("nwait_target: $nwait_target")
        println(collect(zip(dfj.compute_order, dfj.njobs)))
        println("straggler_fraction: $straggler_fraction")
        println()

        # if size(dfj, 1) >= 3
        #     p, coeffs = fit_polynomial(dfj.order, dfj.mean, 3)
        #     ts = range(0, nwait_target, length=100)
        #     # plt.plot(ts, p.(ts))
        #     # println(coeffs)   
        # end     
    end

    # iteration latency for different nwait_target
    # dfi = mean_latency_df(df)
    # plt.plot(dfi.nwait, dfi.latency, ".")
    dfj = by(df, :nwait, :latency => mean => :mean)    
    plt.plot(dfj.nwait ./ nworkers, dfj.mean, "s", label="Iteration latency")
    
    # p, coeffs = fit_polynomial(dfj.nwait, dfj.t_compute_mean, 3)
    # ts = range(0, nworkers, length=100)
    # plt.plot(ts, p.(ts))
    # println(coeffs)    

    plt.legend()
    plt.grid()
    plt.xlabel("Order")
    plt.ylabel("Latency [s]")
    return    
end

# - The latency of each worker at a partiular point of time is characterized by a random variable with some distribution
# - The parameters of that distribution are in turn drawn from some other distribution
# - The parameters of that distribution may change over time, which is captured by the autocorrelation of those parameters
# - Latency may be correlated between workers
# - The latency of a particular iteration is equal to the latency of the w-th fastest worker
# - Which is captured by the w-th order statistic of the random variables characterizing the latency of the individual workers
# - To sample from the iteration latency distribution I need to first generate a set of distributions corresponding to the workers, then draw a sample from each of those distributions, and finally take the w-th largest value of out those samples

function plot_orderstats_flops(df, nworkers=18; onlycompute=true)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency    
    df = df[df.nworkers .== nworkers, :]
    println("worker_flops:\t$(unique(df.worker_flops))")
    println("nbytes:\t$(unique(df.nbytes))\n")
    dfo = orderstats_df(df)    

    # latency order stats recorded for individual workers
    plt.figure()
    nwait_target = nworkers
    for worker_flops in sort!(unique(dfo.worker_flops))
        dfi = dfo
        dfi = dfi[dfi.nwait .== nwait_target, :]
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        # dfi = dfi[dfi.order .<= dfi.nwait, :]       
        # plt.plot(dfi.order, dfi.worker_latency, ".", label="w_target: $nwait_target")
        straggler_fraction = sum(dfi.isstraggler) / length(dfi.isstraggler)
        dfi = dfi[dfi.isstraggler .== false, :]

        dfj = by(dfi, order_col, :worker_latency => mean => :mean, :jobid => ((x) -> length(unique(x))) => :njobs)
        sort!(dfj, order_col)
        plt.plot(dfj[order_col], dfj.mean, "o", label="w_target: $nwait_target ($worker_flops)")

        println("nwait_target: $nwait_target")
        println(collect(zip(dfj[order_col], dfj.njobs)))
        println("straggler_fraction: $straggler_fraction")
        println()

        if size(dfj, 1) >= 3
            p, coeffs = fit_polynomial(dfj[order_col], dfj.mean, 3)
            ts = range(0, nwait_target, length=100)
            # plt.plot(ts, p.(ts))
            # println(coeffs)   
        end     

        # iteration latency for different nwait_target
        dfi = df
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        # dfi = mean_latency_df(df)
        # plt.plot(dfi.nwait, dfi.latency, ".")    
        dfj = by(dfi, :nwait, :latency => mean => :mean)
        plt.plot(dfj.nwait, dfj.mean, "s", label="Iteration latency ($worker_flops)")
        
        # p, coeffs = fit_polynomial(dfj.nwait, dfj.t_compute_mean, 3)
        # ts = range(0, nworkers, length=100)
        # plt.plot(ts, p.(ts))
        # println(coeffs)    


        # latency predicted by the order statistics of Normal random variables
        meanp = Polynomial([2.3552983559702727e-17, 3.5452942951744024e-9, 6.963505495725266e-19])
        varp = Polynomial([6.3248412362377695e-22, 9.520417443453858e-14, 3.2099667366421632e-21])
        μ = meanp(worker_flops)
        σ = sqrt(varp(worker_flops))
        samples = zeros(1000)
        vs = zeros(0)
        for i in 1:nworkers
            s = OrderStatistic(Normal(μ, σ), i, nworkers)
            Distributions.rand!(s, samples)
            push!(vs, mean(samples))
        end
        plt.plot(1:nworkers, vs, "--")
    end

    plt.legend()
    plt.grid()
    plt.xlabel("Order")
    plt.ylabel("Latency [s]")
    return  
end

"""

Fix worker_flops
Plot the difference in latency due to waiting for 1 more worker
"""
function plot_orderstats_derivative(df, worker_flops; onlycompute=true)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    df = df[isapprox.(df.worker_flops, worker_flops, rtol=1e-2), :]
    dfo = orderstats_df(df)
    dfo = dfo[dfo.order .<= dfo.nwait, :]
    dfo.order = dfo.order ./ dfo.nworkers
    dfo.compute_order = dfo.compute_order ./ dfo.nworkers

    plt.figure()
    for nworkers in sort!(unique(dfo.nworkers))
        dfi = dfo
        dfi = dfi[dfi.nworkers .== nworkers, :]
        # return dfi

        dfj = by(dfi, order_col, latency_col => mean => :mean, :jobid => ((x) -> length(unique(x))) => :njobs)
        sort!(dfj, order_col)
        ys = diff(dfj.mean)
        plt.plot(dfj[order_col][1:end-1], ys, "-o", label="Nn: $nworkers")
    end
    plt.grid()
    plt.legend()
    plt.xlabel("w")
    plt.ylabel("Diff")
    return

    # dfo.npartitions = dfo.nworkers .* dfo.nsubpartitions
    df.order = df.order ./ df.nworkers

    # vs = [0.05140742522974717, 0.05067826288956093, 0.05122862096280494, 0.050645562535343865, 0.05099657254820253, 0.05138775739534186, 0.050901895214905575, 0.050316781109837165, 0.05188810059429005, 0.051392694196391135, 0.05082197380306366, 0.050276556499358506, 0.05169503754425293, 0.051862277178989835, 0.05151591182965395, 0.05201310404786811, 0.05083756996715019, 0.05134404845100367]    
    # vs ./= maximum(vs)
    # df.latency ./= vs[df.worker_index]

    dfi = df
    dfi = dfi[dfi.nworkers .== 18, :]
    dfi = dfi[dfi.nsubpartitions .== 2, :]
    # dfi.order = dfi.order ./ dfi.nworkers
    dfj = by(dfi, :order, :worker_latency => mean => :mean)
    sort!(dfj, :order)
    plt.plot(dfj.order[1:end-1], diff(dfj.mean), "o", label="(18, 2)")

    dfi = df
    dfi = dfi[dfi.nworkers .== 36, :]
    dfi = dfi[dfi.nsubpartitions .== 1, :]
    # dfi.order = dfi.order ./ dfi.nworkers    
    dfj = by(dfi, :order, :worker_latency => mean => :mean)
    # plt.plot(dfj.order, dfj.latency_mean, "o", label="(36, 1)")    
    plt.plot(dfj.order[1:end-1], diff(dfj.mean), "s", label="(36, 1)")

    plt.grid()
    plt.ylim(0)
    plt.xlim(0, 1)
    plt.xlabel("order / nworkers")
    plt.ylabel("Diff. latency. wrt. order.")
    plt.legend()
    return  
end

"""

Plot the distribution of the average worker latency
"""
function plot_worker_latency_moments(dfo; miniterations=1000, onlycompute=true, intervalsize=100)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    dfo = dfo[dfo.niterations .>= miniterations, :]
    # dfo = orderstats_df(df)
    dfo = dfo[dfo.order .<= dfo.nwait, :]
    dfo.interval = ceil.(Int, dfo.iteration ./ intervalsize)
    dfi = by(
        dfo, [:jobid, :worker_index, :interval, :worker_flops],
        latency_col => mean => :mean, 
        latency_col => var => :var,
        latency_col => minimum => :minimum,        
        )

    meanp = Polynomial([3.020008104166731e-17, 2.8011867905401972e-9, 3.5443625816981855e-18]) # for < 7e8
    # meanp = Polynomial([2.3552983559702727e-17, 3.5452942951744024e-9, 6.963505495725266e-19])
    varp = Polynomial([6.3248412362377695e-22, 9.520417443453858e-14, 3.2099667366421632e-21])

    # worker latency mean
    plt.figure()
    for worker_flops in sort!(unique(dfi.worker_flops))
        dfj = dfi
        dfj = dfj[dfj.worker_flops .== worker_flops, :]

        # empirical cdf
        xs = sort(dfj.mean)
        ys = 1 .- range(0, 1, length=length(xs))    
        plt.semilogy(xs, ys, label="c: $worker_flops")
        
        # normal distribution cdf fitted to the data
        rv = Distributions.fit(Normal, xs)
        xs = range(0.9*minimum(xs), 1.1*maximum(xs), length=100)    
        plt.semilogy(xs, 1 .- cdf.(rv, xs), "k--")

        # normal distribution predicted by the model
        # μ = meanp(worker_flops)
        # σ = sqrt(varp(worker_flops))
        # plt.plot(xs, cdf.(Normal(μ, σ), xs), "-.")

        # println(((μ, σ), params(rv)))
    end
    plt.legend()
    plt.grid()    
    plt.xlabel("Mean")

    # worker latency variance
    plt.figure()
    for worker_flops in sort!(unique(dfi.worker_flops))
        dfj = dfi
        dfj = dfj[dfj.worker_flops .== worker_flops, :]

        # empirical cdf
        xs = sort(dfj.var)
        ys = range(0, 1, length=length(xs))    
        plt.plot(xs, ys, ".", label="c: $worker_flops")

        # exponential distribution fit to the data
        # rv = Distributions.fit(Exponential, xs)
        # ts = range(0, maximum(xs), length=100)
        # plt.semilogy(ts, 1 .- cdf.(rv, ts), "--")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Variance")

    # minimum worker latency
    plt.figure()
    for worker_flops in sort!(unique(dfi.worker_flops))
        dfj = dfi
        dfj = dfj[dfj.worker_flops .== worker_flops, :]

        # empirical cdf
        xs = sort(dfj.minimum)
        ys = range(0, 1, length=length(xs))    
        plt.plot(xs, ys, label="c: $worker_flops")

        # normal distribution cdf fitted to the data
        rv = Distributions.fit(Normal, xs)
        xs = range(0.9*minimum(xs), 1.1*maximum(xs), length=100)    
        plt.plot(xs, cdf.(rv, xs), "k--")

        # exponential distribution fit to the data
        # rv = Distributions.fit(Exponential, xs)
        # ts = range(0, maximum(xs), length=100)
        # plt.semilogy(ts, 1 .- cdf.(rv, ts), "--")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Minimum")    

    # minimum latency vs. mean
    plt.figure()
    for worker_flops in sort!(unique(dfi.worker_flops))
        dfj = dfi
        dfj = dfj[dfj.worker_flops .== worker_flops, :]
        plt.plot(dfj.mean, dfj.minimum, ".", label="c: $worker_flops")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Mean")    
    plt.ylabel("Minimum")

    return

    # worker latency mean vs. variance
    plt.figure()
    for worker_flops in sort!(unique(dfi.worker_flops))
        dfj = dfi
        dfj = dfj[dfj.worker_flops .== worker_flops, :]
        plt.plot(dfj.mean, dfj.var, ".", label="c: $worker_flops")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Mean")    
    plt.ylabel("Variance")

    return

    dfj = by(dfi, :worker_flops, :mean => mean => :mean, :mean => var => :var)

    dfj = dfj[dfj.worker_flops .< 7e8, :] # TODO: remove

    plt.figure()
    plt.plot(dfj.worker_flops, dfj.mean, ".")
    
    p, coeffs = fit_polynomial(dfj.worker_flops, dfj.mean, 2)
    println(coeffs)
    xs = range(0, maximum(dfj.worker_flops), length=100)
    plt.plot(xs, p.(xs))

    plt.xlabel("c")
    plt.ylabel("mean")
    plt.grid()

    plt.figure()
    plt.plot(dfj.worker_flops, dfj.var, ".")

    p, coeffs = fit_polynomial(dfj.worker_flops, dfj.var, 2)
    println(coeffs)    
    xs = range(0, maximum(dfj.worker_flops), length=100)
    plt.plot(xs, p.(xs))

    plt.xlabel("c")
    plt.ylabel("var")    
    plt.grid()

    return
end

# """


# """
# function foo(df; onlycompute=true)
#     dfo = orderstats_df(df)
#     dfo = dfo[dfo.order .<= dfo.nwait, :]
#     dfi = by(dfo, [:jobid, :worker_index], latency_col => mean => :mean)    
#     order_col = onlycompute ? :compute_order : :order
#     latency_col = onlycompute ? :worker_compute_latency : :worker_latency
# end

"""

Return a df composed of the parameters of a Gamma distribution fit to the latency of each 
individual worker for each job.
"""
function gamma_model_df(df; miniterations=0)
    df = df[.!ismissing.(df["latency_worker_1"]), :]
    df = df[df.nwait .== df.nworkers, :]
    if size(df, 1) == 0
        return DataFrame()
    end
    nworkers = maximum(df.nworkers)

    # use to verify the values computed by the "by" setup
    # jobid = 242
    # dfi = df
    # dfi = dfi[dfi.jobid .== jobid, :]
    # vs = float.(dfi.latency_worker_1)
    # shape, scale = params(Distributions.fit(Gamma, vs))
    # println("shape: $shape, scale: $scale")

    df = stack(df, ["latency_worker_$i" for i in 1:nworkers])
    df = dropmissing(df, :value)
    dfi = by(
        df, [:jobid, :variable], 
        :nworkers => mean => :nworkers,
        :worker_flops => mean => :worker_flops,
        :iteration => maximum => :niterations,
        :value => mean => :mean,
        :value => var => :var,
    )    
    df = by(
        df, [:jobid, :variable], 
        :value => ((x) -> NamedTuple{(:shape, :scale)}(params(Distributions.fit(Gamma, x)))),        
    )    
    df[:gmean] = df.shape .* df.scale # mean of fitted Gamma
    df[:gvar] = df.shape .* df.scale.^2 # variance of fitted Gamma
    df[:mean] = dfi[:mean] # empirical mean
    df[:var] = dfi[:var] # empirical variance    
    df[:nworkers] = dfi[:nworkers]
    df[:worker_flops] = dfi[:worker_flops]
    df[:niterations] = dfi[:niterations]

    # drop rows with too few samples (iterations)
    df = df[df.niterations .>= miniterations, :]

    # avoid rounding errors when splitting by worker_flops later
    df.worker_flops .= round.(df.worker_flops, digits=6)

    sort!(df, :jobid)
end

"""

Plot the latency of individual workers from fastest to slowest.
"""
function plot_worker_latency(df, nworkers=12, nsubpartitions=5)    
    df = df[df.nworkers .== nworkers, :]
    df = df[df.nsubpartitions .== nsubpartitions, :]
    df = df[df.nwait .== df.nworkers, :]
    df = gamma_model_df(df)
    if size(df, 1) == 0
        println("No jobs")
        return
    end
    plt.figure()
    for jobid in sort!(unique(df.jobid))
        println(jobid)
        dfi = df
        dfi = dfi[dfi.jobid .== jobid, :]        
        xs = sort!(dfi.mean)
        plt.plot(xs, ".-", label="ID: $jobid")
    end
    plt.grid()
    plt.legend()
    plt.xlabel("w")
    plt.ylabel("Mean latency")
    return
end

"""

Plot the parameters of a Gamma distribution fit to the latency of individual workers.
"""
function plot_gamma_model(df)
    dfm = gamma_model_df(df)
    # dfm = dfm[dfm.nworkers .== 18, :]

    # plot mean
    plt.figure()
    for nworkers in sort!(unique(dfm.nworkers))
        dfi = dfm
        dfi = dfi[dfi.nworkers .== nworkers, :]                
        plt.plot(dfi.worker_flops, dfi.gmean, ".", label="Nn: $nworkers (Gamma)")
        # plt.plot(dfi.worker_flops, dfi.mean, ".", label="Nn: $nworkers (emp.)")
        dfj = by(dfi, :worker_flops, :mean => mean => :mean)
        plt.plot(dfj.worker_flops, dfj.mean, "s")
    end
    plt.title("Mean")
    plt.ylabel("Mean")
    plt.grid()
    plt.legend()

    # plot variance
    plt.figure()
    for nworkers in sort!(unique(dfm.nworkers))
        dfi = dfm
        dfi = dfi[dfi.nworkers .== nworkers, :]                
        plt.plot(dfi.worker_flops, dfi.gvar, ".", label="Nn: $nworkers (Gamma)")
        # plt.plot(dfi.worker_flops, dfi.var, ".", label="Nn: $nworkers (emp.)")
        dfj = by(dfi, :worker_flops, :var => mean => :var)
        plt.plot(dfj.worker_flops, dfj.var, "s")
    end
    plt.title("Variance")
    plt.ylabel("Variance")
    plt.grid()
    plt.legend()    

    # plot scale
    plt.figure()
    for nworkers in sort!(unique(dfm.nworkers))
        dfi = dfm
        dfi = dfi[dfi.nworkers .== nworkers, :]                
        plt.plot(dfi.worker_flops, dfi.scale, ".", label="Nn: $nworkers")
        dfj = by(dfi, :worker_flops, :scale => mean)
        plt.plot(dfj.worker_flops, dfj.scale_mean, "o")
    end
    plt.title("Scale")
    plt.ylabel("Scale")
    plt.grid()
    plt.legend()
    return    

    # # plot scale as a function of mean
    # plt.figure()
    # for nworkers in sort!(unique(dfm.nworkers))
    #     dfi = dfm
    #     dfi = dfi[dfi.nworkers .== nworkers, :]                
    #     plt.plot(dfi.mean, dfi.scale, ".", label="Nn: $nworkers")
    #     # dfj = by(dfi, :worker_flops, :scale => mean)
    #     # plt.plot(dfj.worker_flops, dfj.scale_mean, "o")
    # end
    # plt.title("Scale")
    # plt.xlabel("Gamma mean")    
    # plt.ylabel("Scale")
    # plt.grid()
    # plt.legend()    

    # plot shape
    plt.figure()
    for nworkers in sort!(unique(dfm.nworkers))
        dfi = dfm
        dfi = dfi[dfi.nworkers .== nworkers, :]
        plt.semilogy(dfi.worker_flops, dfi.shape, ".", label="Nn: $nworkers")
        dfj = by(dfi, :worker_flops, :shape => mean)
        plt.semilogy(dfj.worker_flops, dfj.shape_mean, "o")        
    end
    plt.ylabel("Shape")    
    plt.grid()
    plt.legend()    

end

function plot_worker_cdf(df, n=1; miniterations=10000, onlycompute=true)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    df = df[df.niterations .>= miniterations, :]
    # df = orderstats_df(df)

    # two-state: 602, 34
    # single-state: 600, 17    

    

    plt.figure()
    for i in 1:n

        # if i == 1
        #     jobid = 600
        #     worker_index = 17
        # elseif i == 2
        #     jobid = 602
        #     worker_index = 34
        # end

        jobid = 609
        worker_index = 13

        dfi = df
        # jobid = rand(unique(dfi.jobid))
        dfi = dfi[dfi.jobid .== jobid, :]
        # worker_index = rand(unique(dfi.worker_index))
        dfi = dfi[dfi.worker_index .== worker_index, :]
        
        # empirical cdf
        xs = sort!(dfi[latency_col])
        shift = minimum(xs) - eps(Float64)
        xs .-= shift
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, "b-", label="Empirical")

        # Gamma
        rv = Distributions.fit(Gamma, float.(xs))
        ts = range(0.9*minimum(xs), 1.1*maximum(xs), length=100)
        plt.plot(ts, cdf.(rv, ts), "k--", label="Gamma")
        println(params(rv))

        # LogNormal
        # rv = Distributions.fit(LogNormal, float.(xs))
        # ts = range(0.9*minimum(xs), 1.1*maximum(xs), length=100)
        # plt.semilogy(ts, 1 .- cdf.(rv, ts), "r--", label="LogNormal")   

        # # Exponential
        # rv = Distributions.fit(Exponential, float.(xs))
        # ts = range(0.9*minimum(xs), 1.1*maximum(xs), length=100)        
        # plt.semilogy(ts, 1 .- cdf.(rv, ts), "m-.")

        # Erlang
        # rv = Distributions.fit(Erlang, float.(xs))
        # ts = range(0.9*minimum(xs), 1.1*maximum(xs), length=100)
        # plt.semilogy(ts, 1 .- cdf.(rv, ts), "m--", label="Erlang")             

        # Normal
        # rv = Distributions.fit(Normal, float.(xs))
        # ts = range(0.9*minimum(xs), 1.1*maximum(xs), length=100)
        # plt.semilogy(ts, 1 .- cdf.(rv, ts), "m-")        
        
    end
    # plt.ylim(1e-1, 1)
    plt.legend()
    plt.grid()
    plt.ylabel("CCDF")
    plt.xlabel("Latency [s]")
    return
end

function plot_worker_latency_timeseries(df, n=3; miniterations=10000, onlycompute=true, worker_flops=nothing, intervalsize=100)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    df = df[df.niterations .>= miniterations, :]
    if !isnothing(worker_flops)
        df = df[isapprox.(df.worker_flops, worker_flops, rtol=1e-2), :]
    end
    df.interval = ceil.(Int, df.iteration ./ intervalsize)
    df = by(
        df, [:jobid, :worker_index, :interval, :worker_flops],
        latency_col => mean => :mean, 
        latency_col => median => :median,
        latency_col => var => :var,
        latency_col => minimum => :minimum,        
        )

    plt.figure()
    for _ in 1:n
        dfi = df
        jobid = rand(unique(dfi.jobid))
        dfi = dfi[dfi.jobid .== jobid, :]
        worker_index = rand(unique(dfi.worker_index))
        dfi = dfi[dfi.worker_index .== worker_index, :]
        plt.plot(dfi.mean, ".", label="job: $jobid, worker: $worker_index")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Iteration index")
    plt.ylabel("Latency")
end

function plot_worker_latency_process(dfo, n=10; miniterations=10000, onlycompute=true, worker_flops=nothing, intervalsize=1)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    dfo = dfo[dfo.niterations .>= miniterations, :]
    if !isnothing(worker_flops)
        dfo = dfo[isapprox.(dfo.worker_flops, worker_flops, rtol=1e-2), :]
    end
    dfo.interval = ceil.(Int, dfo.iteration ./ intervalsize)
    dfo = by(
        dfo, [:jobid, :worker_index, :interval, :worker_flops],
        latency_col => mean => :mean, 
        latency_col => median => :median,
        latency_col => var => :var,
        latency_col => minimum => :minimum,        
        )

    plt.figure()
    for worker_flops in sort!(unique(dfo.worker_flops))
        dfi = dfo
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        dfi = by(dfi, [:jobid, :worker_index], :mean => diff => :diff)
        xs = sort!(dfi.diff)
        ys = 1 .- range(0, 1, length=length(xs))
        # plt.semilogy(xs, ys, label="c: $worker_flops")

        # Normal
        # rv = Distributions.fit(Normal, xs)
        # ts = range(0.9*minimum(xs), 1.1*maximum(xs), length=500)
        # plt.plot(ts, cdf.(rv, ts), "k--")

        # Cauchy
        # rv = Distributions.fit(Normal, xs)
        # ts = range(1.1*minimum(xs), 1.1*maximum(xs), length=1000)
        # plt.semilogy(ts, 1 .- cdf.(rv, ts), "r--")        

        # Tukey-Lambda
        for λ in [-0.4]
            rv = TukeyLambda(λ)

            qs = range(0.05, 0.95, length=100)
            xs = [quantile(xs, q) for q in qs]
            xs ./= maximum(abs.(xs))
            ys = [quantile(rv, q) for q in qs]
            ys ./= maximum(abs.(ys))
            plt.plot(xs, ys, label="λ: $λ")        

            xs = quantile.(rv, [0.01, 0.99])
            xs ./= maximum(abs.(xs))
            ys = quantile.(rv, [0.01, 0.99])
            ys ./= maximum(abs.(ys))
            plt.plot(xs, ys, "k-")                    
        end
        # plt.plot([-1, 1], [-1, 1], "k-")
    end
    plt.xlabel("Data")
    plt.ylabel("Theoretical distribution")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    return

    plt.figure()
    for _ in 1:n
        dfi = dfo
        jobid = rand(unique(dfi.jobid))
        dfi = dfi[dfi.jobid .== jobid, :]
        worker_index = rand(unique(dfi.worker_index))
        dfi = dfi[dfi.worker_index .== worker_index, :]
        plt.plot(diff(dfi.mean), ".", label="job: $jobid, worker: $worker_index")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Iteration index")
    plt.ylabel("Latency")
end