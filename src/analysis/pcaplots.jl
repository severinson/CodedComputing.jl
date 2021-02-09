using CSV, DataFrames, PyPlot, Statistics, Polynomials
using StatsBase

using PyCall
tikzplotlib = pyimport("tikzplotlib")

# linear model
get_βs() = [0.005055059937837611, 8.075122937312302e-8, 1.1438758464435006e-16]
get_γs() = [0.03725188744901591, 3.109510011653974e-8, 6.399147477943208e-16]
get_offset(w) = 0.005055059937837611 .+ 8.075122937312302e-8w .+ 1.1438758464435006e-16w.^2
get_slope(w, nworkers) = 0.03725188744901591 .+ 3.109510011653974e-8(w./nworkers) .+ 6.399147477943208e-16(w./nworkers).^2

# shifted exponential model
get_shift(w) = 0.2514516116132241 .+ 6.687583396247953e-8w .+ 2.0095825408761404e-16w.^2
get_scale(w) = 0.23361469930191084 .+ 7.2464826067975726e-9w .+ 5.370433628859458e-17w^2

"""

For each job, replace the delay of the first iteration by the average delay of the remaining iterations.
Since we are not interested in the time spent on initialization in the first iteration.
"""
function remove_initialization_delay!(df)
    jobids = unique(df.jobid)
    t_update = zeros(length(jobids))
    t_compute = zeros(length(jobids))
    for (i, jobid) in enumerate(jobids)
        dfi = df
        dfi = dfi[dfi.jobid .== jobid, :]
        dfi = dfi[dfi.iteration .>= 2, :]
        t_update[i] = mean(dfi.t_update)
        t_compute[i] = mean(dfi.t_compute)
    end
    df[df.iteration .== 1, "t_update"] .= t_update

    # don't do it for t_compute, since workers are properly ininitialized anyway, and the averaging messes with kickstart
    # df[df.iteration .== 1, "t_compute"] .= t_compute

    df
end

"""

Return t_compute, computed from the model fit to the 1000 genomes chromosome 20 data.
Set `samp` to a value larger than 1 to increase the effect of straggling, and to a value in [0, 1) to reduce the effect.
"""
function model_tcompute_from_df(df; samp=1.0)
    
    # t_compute for everything but kickstart iterations
    rv = get_offset.(df.worker_flops) .+ samp.*get_slope.(df.worker_flops, df.nworkers) .* df.nwait

    # handle kickstart
    mask = df.kickstart .== true .& df.iteration .== 1
    rv[mask] .= get_offset.(df[mask, :worker_flops]) .+ samp.*get_slope.(df[mask, :worker_flops], df[mask, :nworkers]) .* df[mask, :nworkers]

    return rv
end

"""

Return a vector composed of the cumulative compute time for each job.
"""
function cumulative_time_from_df(df)
    sort!(df, [:jobid, :iteration])
    rv = zeros(size(df, 1))
    for jobid in unique(df.jobid)
        mask = df.jobid .== jobid
        df_jobid = df[mask, :]
        sort!(df_jobid, "iteration")
        rv[mask] .= cumsum(df_jobid.t_compute .+ df_jobid.t_update)
    end
    rv
end

"""

Return a vector composed of the number of flops performed by each worker and iteration.

1000 genomes, chr20 density: 0.05117854232324947
"""
function worker_flops_from_df(df; density=0.05117854232324947)
    nflops = float.(df.nrows)
    nflops ./= df.nworkers
    nflops .*= df.nreplicas
    nflops .*= Missings.replace(df.pfraction, 1.0)
    nflops ./= Missings.replace(df.nsubpartitions, 1.0)
    nflops .*= 2 .* df.ncolumns .* df.ncomponents
    nflops .*= density
end

"""

Return a vector composed of the number of floats communicated per iteration.
"""
function communication_from_df(df)
    2 .* df.ncolumns .* df.ncomponents
end

function read_df(filename="C:/Users/albin/Dropbox/Eigenvector project/data/dataframes/210208/210208_v4.csv"; nworkers=nothing)
    df = DataFrame(CSV.File(filename, normalizenames=true))
    df[:nostale] = Missings.replace(df.nostale, false)
    df[:kickstart] = Missings.replace(df.kickstart, false)
    df = df[.!ismissing.(df.nworkers), :]
    # df = df[df.nostale .== false, :]
    df = df[df.kickstart .== false, :]
    if !isnothing(nworkers)
        df = df[df.nworkers .== nworkers, :]
    end
    df = remove_initialization_delay!(df)
    df[:worker_flops] = worker_flops_from_df(df)

    # scale up workload
    # df[:worker_flops] .*= 22
    # df[:t_compute] .= model_tcompute_from_df(df, samp=1)

    df[:communication] = communication_from_df(df)
    df[:t_total] = cumulative_time_from_df(df)    
    df, split_df_by_algorithm(df)
end

function split_df_by_algorithm(df)
    rv = Dict{String,DataFrame}()
    for (algo, variancereduced) in Iterators.product(unique(df.algorithm), unique(df.variancereduced))
        if ismissing(variancereduced)
            continue
        end
        dfi = df
        dfi = dfi[dfi.algorithm .== algo, :]        
        if algo == "pca.jl"
            dfi = dfi[dfi.variancereduced .== variancereduced, :]
        end
        if size(dfi, 1) == 0
            continue
        end
        label = algo[1:end-3]
        if variancereduced
            label *= "vr"
        end
        rv[label] = dfi
    end
    rv
end

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
    # c3 = 2*βs[3]*σ0^2
    # c4 = 3*γs[3]*f*σ0^2
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
    poly = Polynomials.fit(float.(dfm.worker_flops), float.(dfm.offset), 2)    
    t = range(0, maximum(dfm.worker_flops), length=100)
    plt.semilogx(t, poly.(t))

    # print fit line
    println(poly.coeffs)
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
    poly = Polynomials.fit(float.(dfm.worker_flops ./ dfm.nworkers), float.(dfm.slope), 2)    
    t = range(0, maximum(dfm.worker_flops ./ dfm.nworkers), length=100)
    plt.semilogx(t, poly.(t))

    # print fit line
    println(poly.coeffs)
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
        # return float.(dfi.nwait), float.(dfi.t_compute)
        poly = Polynomials.fit(float.(dfi.nwait), float.(dfi.t_compute), 1)
        xs = [0, nworkers]        
        plt.plot(xs, poly.(xs), "--", label="Local model")            
        println(poly)

        # offset, slope = linear_model(float.(dfi.nwait), float.(dfi.t_compute))
        # ys = offset .+ slope .* xs
        # plt.plot(xs, ys, "--", label="Local model")    

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

"""

Return a matrix of size `niterations` by `nworkers`, where a `1` in position `i, j` indicates that
worker `j` was a straggler in iteration `i`. Non-stragglers are marked with `-1`.
"""
function straggler_matrix_from_jobid(df, jobid, nworkers)
    df = df[df.jobid .== jobid, :]
    sort!(df, :iteration)
    rv = fill(-1, size(df, 1), nworkers)
    for i in 1:nworkers
        rv[df["repoch_worker_$(i)"] .< df.iteration, i] .= 1
    end
    ts = zeros(size(df, 1))
    ts[2:end] .= df.t_total[1:end-1]
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
function plot_staleness(df, nworkers=18, nwait=1, nsubpartitions=1)
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.pfraction .== 1, :]    
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
        if !ismissing(dfi.mse[1])
            continue
        end
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

        # traces with mse are recorded in a less controlled manner
        # so we skip these
        if !ismissing(dfi.mse[1])
            continue
        end

        @assert length(unique(df.nworkers)) == 1
        nworkers = unique(df.nworkers)[1]

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
    println("Count: $counts")
    return edges, values, counts
end

"""

Plot the probability that a straggler remains a straggler after some time has passed.
"""
function plot_straggler_ac(df; f=0.5)
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.pfraction .== 1, :]    
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
    for nworkers in [6, 12, 18]
        nwait = round(Int, nworkers*f)
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        dfi = dfi[dfi.nwait .== nwait, :]
        nbins = round(Int, maximum(dfi.t_total) / 10)
        edges, values, counts = straggler_prob_timeseries_from_df(dfi; nbins)
        plt.plot(edges[1:end-1], values, ".-", label="Nn: $nworkers")

        # print values
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

Plot the best error over all 
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

"""

Fit a linear model (i.e., a line) to the data X, y.
"""
function linear_model(X::AbstractMatrix, y::AbstractVector)
    size(X, 1) == length(y) || throw(DimensionMismatch("X has dimensions $(size(X)), but y has dimension $(length(y))"))
    A = ones(size(X, 1), size(X, 2)+1)    
    A[:, 2:end] .= X
    A \ y
end

linear_model(x::AbstractVector, y::AbstractVector) = linear_model(reshape(x, length(x), 1), y)

function write_table(xs::AbstractVector, ys::AbstractVector, filename::AbstractString)
    length(xs) == length(ys) || throw(DimensionMismatch("xs has dimension $(length(xs)), but ys has dimension $(length(ys))"))
    open(filename, "w") do io
        for i in 1:length(xs)
            write(io, "$(xs[i]) $(ys[i])\n")
        end
    end
    return
end

# Let's figure out if the straggling is explained by the varying density
# First, let's check if the workers storing more dense partitions are consistently the slowest

function plot_straggler_density(df)
    nworkers = 18
    nsubpartitions = 1
    nwait = 9
    dfi = df
    dfi = dfi[dfi.nworkers .== nworkers, :]
    dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
    dfi = dfi[dfi.nwait .== nwait, :]
    dfi = dfi[dfi.nreplicas .== 1, :]
    dfi = dfi[dfi.pfraction .== 1, :]

    # for original data matrix
    vs = [0.04688062471391655, 0.04749676843908486, 0.04941002489203347, 0.04862445444082279, 0.048793894574690695, 0.04943647884363674, 0.051972915153795755, 0.05602539385691622, 0.059106140262184935, 0.05671962158324654, 0.04848918053618723, 0.04840049669933181, 0.04903849489666922, 0.0502636980261616, 0.05460620642087488, 0.056018742668364305, 0.05186742079503943,
    0.04808694236837983]

    # for shuffled data matrix
    vs = [0.05140742522974717, 0.05067826288956093, 0.05122862096280494, 0.050645562535343865, 0.05099657254820253, 0.05138775739534186, 0.050901895214905575, 0.050316781109837165, 0.05188810059429005, 0.051392694196391135, 0.05082197380306366, 0.050276556499358506, 0.05169503754425293, 0.051862277178989835, 0.05151591182965395, 0.05201310404786811, 0.05083756996715019, 0.05134404845100367]
    vs ./= maximum(vs)

    # foo = zeros(maximum(dfi.iteration))
    # bar = zeros(maximum(dfi.iteration))

    for jobid in unique(dfi.jobid)
        dfj = dfi
        dfj = dfj[dfj.jobid .== jobid, :]
        if ismissing(dfj[1, "repoch_worker_1"])
            continue
        end
        M, ts = straggler_matrix_from_jobid(dfi, jobid, nworkers)
        replace!(M, -1 => 0)
        
        foo = round.(sum(M, dims=1) ./ size(M, 1) .+ eps(Float64), digits=3)
        bar = round.(vs, digits=3)

        p = sortperm(bar)
        foo = foo[p]
        bar = bar[p]
        for i in 1:length(foo)
            println("$(foo[i])\t$(bar[i])")
        end    
        println("===========")        
        println()

        # return M
    end


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
    end

    df = df[df.nworkers .== nworkers, :]    
    df = df[df.kickstart .== false, :]
    # df = df[df.nostale .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[.!ismissing.(df.mse), :]

    # plot the bound
    r = 2
    Nw = 1
    samp = 1

    # get the convergence per iteration for batch GD
    dfi = df
    dfi = dfi[dfi.nsubpartitions .== 1, :]    
    dfi = dfi[dfi.nwait .== nworkers, :]
    dfi = dfi[dfi.stepsize .== 1, :]
    dfi = dfi[dfi.variancereduced .== false, :]
    dfi = dfi[dfi.kickstart .== false, :]
    dfi = dfi[dfi.nostale .== false, :]
    dfj = combine(groupby(dfi, :iteration), :mse => mean)
    ys = opt .- dfj.mse_mean

    # compute the iteration time for a scheme with a factor r replication
    @assert length(unique(dfi.worker_flops)) == 1
    worker_flops = r*unique(dfi.worker_flops)[1]
    x0 = get_offset(worker_flops) .+ samp .* get_slope(worker_flops, nworkers) * Nw
    xs = x0 .* (1:maximum(dfi.iteration))

    # make the plot
    plt.figure()        
    plt.semilogy(xs, ys, "--k", label="Bound r: $r, Nw: $Nw")
    write_table(xs, ys, "./data/bound_$(nworkers)_$(Nw)_$(r).csv")

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
            dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)
            if size(dfj, 1) > 0
                xs = dfj.t_total_mean
                ys = opt.-dfj.mse_mean
                plt.semilogy(xs, ys, "o-", label="DSAG (Nw: $nwait, Np: $nsubpartitions, η: $stepsize)")                
                write_table(xs, ys, filename)
            end
        end

        ### SAG
        dfj = dfi
        dfj = dfj[dfj.variancereduced .== true, :]
        if nwait < nworkers # for nwait = nworkers, DSAG and SAG are the same
            dfj = dfj[dfj.nostale .== true, :]            
        end        
        filename = "./data/sag_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"
        println("SAG: $(length(unique(dfj.jobid))) jobs")
        if size(dfj, 1) > 0
            dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)
            if size(dfj, 1) > 0
                xs = dfj.t_total_mean
                ys = opt.-dfj.mse_mean                
                plt.semilogy(xs, ys, "^-", label="SAG (Nw: $nwait, Np: $nsubpartitions, η: $stepsize)")
                write_table(xs, ys, filename)                
            end
        end

        ### SGD
        dfj = dfi
        dfj = dfj[dfj.variancereduced .== false, :]
        filename = "./data/sgd_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"
        println("SGD: $(length(unique(dfj.jobid))) jobs")
        if size(dfj, 1) > 0
            dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)    
            if size(dfj, 1) > 0
                xs = dfj.t_total_mean
                ys = opt.-dfj.mse_mean                
                plt.semilogy(xs, ys, "s-", label="SGD (Nw: $nwait, Np: $nsubpartitions, η: $stepsize)")
                write_table(xs, ys, filename)
            end
        end
        
        println()
    end

    plt.grid()
    plt.legend()    
    plt.xlabel("Time [s]")
    plt.ylabel("Explained Variance Sub-optimality Gap")
    return
end