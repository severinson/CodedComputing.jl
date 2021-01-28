using CSV, DataFrames, PyPlot, Statistics, Polynomials

using PyCall
tikzplotlib = pyimport("tikzplotlib")

# linear model parameters
get_βs() = [0.011292726710870777, 8.053097269359726e-8, 1.1523850574475912e-16]
get_γs() = [0.037446901552194996, 3.1139078757476455e-8, 6.385299464228732e-16]
get_offset(σ) = 0.011292726710870777 .+ 8.053097269359726e-8σ + 1.1523850574475912e-16σ.^2
get_slope(σ, nworkers) = 0.037446901552194996 .+ 3.1139078757476455e-8(σ./nworkers) + 6.385299464228732e-16(σ./nworkers).^2

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

function read_df(filename="C:/Users/albin/Dropbox/Eigenvector project/data/pca/1000genomes/aws12/210114_v13.csv"; nworkers=nothing)
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
function fit_shiftexp_model(df; nworkers, nwait, nsubpartitions=1)
    # df = df[df.nwait .== nwait, :]
    df = df[df.nsubpartitions .== nsubpartitions, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.kickstart .== false, :]    
    for nworkers in sort!(unique(df.nworkers))
        println("Nn: $nworkers")
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]        
        for nwait in sort!(unique(dfi.nwait))
            println("Nw: $nwait")
            dfj = dfi
            dfj = dfj[dfj.nwait .== nwait, :]
            σ = var(df.t_compute)
            μ = mean(df.t_compute) - quantile(df.t_compute, 0.04)
            β1 = sqrt(σ / sum(1/i^2 for i in (nworkers-nwait+1):nworkers))
            β2 = μ / sum(1/i for i in (nworkers-nwait+1):nworkers)
            println((β1, β2))        
        end
    end
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
function plot_predictions(σ0=1.393905852e9)

    nworkers_all = 1:200
    f = 1.0
    σ0s = 10.0.^range(5, 12, length=20)    

    # plot the speedup due to waiting for fewer workers    
    # for fi in [0.1, 0.5]
    #     f1 = fi
    #     f2 = f
    #     nws1 = optimize_nworkers.(σ0s, f1)
    #     nws2 = optimize_nworkers.(σ0s, f2)
    #     ts1 = get_offset.(σ0s./nws1) .+ get_slope.(σ0s./nws1, nws1) .* f1 .* nws1
    #     ts2 = get_offset.(σ0s./nws2) .+ get_slope.(σ0s./nws2, nws2) .* f2 .* nws2
    #     plt.semilogx(σ0s, ts2 ./ ts1, label="f: $fi")
    # end
    # plt.xlabel("σ0")
    # plt.ylabel("speedup")
    # plt.grid()
    # plt.legend()          

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
    return
    
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
    for nsubpartitions in [1/2, 1, 3, 20]
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

        plt.plot([x], ts[round(Int, x)], "o")
        println("Np: $nsubpartitions, x: $x, t: $(ts[round(Int, x)])")        
    end

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

"""
function plot_compute_time(df)
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]

    # plot the slope of the compute time
    for nworkers in sort!(unique(df.nworkers))

        if nworkers == 36
            continue
        end        

        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        xs = Vector{Float64}()
        ys = Vector{Float64}()
        for worker_flops in sort!(unique(dfi.worker_flops))
            dfj = dfi
            dfj = dfj[dfj.worker_flops .== worker_flops, :]
            if size(dfj, 1) == 0
                continue
            end
            dfk = combine(groupby(dfj, :nwait), :t_compute => mean)
            if size(dfk, 1) == 0
                continue
            end
            sort!(dfk, :nwait)
        
            # slope between adjacent points
            x = zeros(size(dfk, 1)-1)
            for i in 2:size(dfk, 1)
                x[i-1] = (dfk.t_compute_mean[i] - dfk.t_compute_mean[i-1]) / (dfk.nwait[i] - dfk.nwait[i-1])
            end
            x .*= nworkers # normalize slope by number of workers

            # plt.plot(fill(worker_flops, length(x)), x, ".", label="$((nworkers, worker_flops))")        
            # plt.plot(worker_flops, mean(x[2:end]), "o")
            # push!(xs, worker_flops)
            # push!(ys, mean(x))

            append!(xs, fill(worker_flops, length(x)))
            append!(ys, x)
        end

        # plot all data points
        plt.plot(xs, ys, "o", label="Nn: $nworkers")

        # plot the average for each value of worker_flops



        # quadratic fit
        if length(xs) > 2
            poly = Polynomials.fit(xs, ys, 2)
            println(poly)
            t = range(0.0, maximum(df.worker_flops), length=100)
            plt.plot(t, poly.(t))        
        end

        # mean fit
        # m = mean(ys)
        # plt.plot([0, maximum(df.worker_flops)], [m, m])
    end    
    plt.xlabel("flops")
    plt.ylabel("Slope between adjacent points")
    plt.grid()
    plt.legend()    
    return

    # plot the time for the first worker to respond
    for nworkers in sort!(unique(df.nworkers))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        dfi = dfi[dfi.nwait .== 1, :]
        if size(dfi, 1) == 0
            continue
        end
        dfj = combine(groupby(dfi, :worker_flops), :t_compute => mean)
        if size(dfj, 1) <= 1
            continue
        end                
        sort!(dfj, :worker_flops)            

        plt.plot(dfj.worker_flops, dfj.t_compute_mean, "o-", label="Nn: $nworkers")

        # quadratic fit
        poly = Polynomials.fit(dfj.worker_flops, dfj.t_compute_mean, 2)
        println(poly)
        t = range(0.0, maximum(dfj.worker_flops), length=100)
        plt.plot(t, poly.(t))
    end
    plt.title("Time until the fastest worker responds")
    plt.xlabel("nflops")
    plt.ylabel("Compute time [s]")
    plt.grid()
    plt.legend()
    return

    # fix nworkers, nwait
    plt.figure()
    for (nworkers, nwait) in Iterators.product(unique(df.nworkers), unique(df.nwait))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        dfi = dfi[dfi.nwait .== nwait, :]
        if size(dfi, 1) == 0
            continue
        end
        dfj = combine(groupby(dfi, :worker_flops), :t_compute => mean)
        if size(dfj, 1) == 0
            continue
        end        
        sort!(dfj, :worker_flops)
        plt.plot(dfj.worker_flops, dfj.t_compute_mean, ".-", label="Nn: $nworkers, Nw: $nwait")
    end
    plt.grid()
    plt.legend()
    plt.xlabel("worker_flops")
    plt.ylabel("Compute time [s]")

    # fix nworkers, worker_flops
    plt.figure()
    for (nworkers, worker_flops) in Iterators.product(unique(df.nworkers), unique(df.worker_flops))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        if size(dfi, 1) == 0
            continue
        end
        dfj = combine(groupby(dfi, :nwait), :t_compute => mean)
        if size(dfj, 1) == 0
            continue
        end        
        sort!(dfj, :nwait)
        plt.plot(dfj.nwait, dfj.t_compute_mean, ".-", label="Nn: $nworkers, flops: $worker_flops")
    end
    plt.grid()
    plt.legend()
    plt.xlabel("Nw")
    plt.ylabel("Compute time [s]")    

    # fix nwait, worker_flops
    plt.figure()
    for (nwait, worker_flops) in Iterators.product(unique(df.nwait), unique(df.worker_flops))
        dfi = df
        dfi = dfi[dfi.nwait .== nwait, :]
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        if size(dfi, 1) == 0
            continue
        end
        dfj = combine(groupby(dfi, :nworkers), :t_compute => mean)
        if size(dfj, 1) == 0
            continue
        end        
        sort!(dfj, :nworkers)
        plt.plot(dfj.nworkers, dfj.t_compute_mean, ".-", label="Nw: $nwait, flops: $worker_flops")
    end
    plt.grid()
    plt.legend()
    plt.xlabel("Nn")
    plt.ylabel("Compute time [s]")    


    # tikzplotlib.save("./plots/tcompute.tex")
    
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

    # plt.figure()
    # for (label, df) in dct
    #     xs = Vector{Float64}()
    #     ys = Vector{Float64}()
    #     for nsubpartitions in unique(df.nsubpartitions)
    #         dfi = df
    #         dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
    #         if size(dfi, 1) == 0
    #             continue
    #         end
    #         push!(xs, nsubpartitions)
    #         push!(ys, mean(dfi.t_update))
    #         l = label * " partitions: $nsubpartitions"
    #         plt.plot(dfi.nsubpartitions, dfi.t_update, ".", label=l)
    #     end
    #     p = sortperm(xs)        
    #     plt.plot(xs[p], ys[p], ".-", label=label)
    # end
    # plt.grid()
    # plt.legend()
    # plt.xlabel("Number of sub-partitions")
    # plt.ylabel("Update time [s]")
    # return    
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

For a given job id, plot traces indicating which workers responded in each iteration.
"""
function plot_response_traces(df; jobid=nothing)
    if isnothing(jobid) || !(jobid in df.jobid)
        println("jobid must be one of:")
        println(unique(df.jobid))
        return
    end
    df = df[df.jobid .== jobid, :]
    nworkers = df.nworkers[1]
    plt.figure()
    for i in 1:nworkers
        x = findall(df["worker_$(i)_responded"])
        y = repeat([i], length(x))
        plt.plot(x, y, "ro")
    end
    plt.title("Markers indicate which workers responded in each iteration")
    plt.ylabel("Worker index")
    plt.xlabel("Iteration")
    plt.xlim(0, size(df, 1))
    plt.ylim(1, nworkers)
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

Plot the MSE at `niterations` against the fraction of the matrix processed per iteration. Plots
results for each unique `stepsize` separately.
"""
function plot_convergence_rate(df, iteration::Integer=10; filters=Dict{String,Any}())
    df = dropmissing(df)
    df = df[df.variancereduced .== true, :]
    df = df[df.iteration .== iteration, :]

    # apply filters
    for (key, value) in filters
        df = df[df[key] .== value, :]
    end    

    # compute what fraction of the matrix was processed in each iteration
    # TODO: doesn't account for replicas
    df[:fraction] = 1 ./ df.nsubpartitions .* df.pfraction
        
    plt.figure()
    for nwait in unique(df.nwait)
        df_nwait = df[df.nwait .== nwait, :]
        for stepsize in unique(df_nwait.stepsize)
            df_stepsize = df_nwait[df_nwait.stepsize .≈ stepsize, :]
            plt.plot(df_stepsize.fraction, df_stepsize.mse, "o", label="nwait $nwait, stepsize $stepsize")
        end
    end
    plt.grid()
    plt.xlabel("Fraction of local data processed per iteration")
    plt.ylabel("Explained variance at iteration $iteration")
    plt.legend()
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

Plot the time (or number of iterations) until the explained variance has converged to within
`atol + rtol*opt` of `opt` as a function of `nsubpartitions` separately for each unique 
(`nworkers`, `nwait`) pair. Use this plot to select optimal values for `nwait` and `nsubpartitions` 
for a given number of workers.
"""
function plot_convergence_time(df; opt=maximum(df.mse), atol=0, rtol=1e-2, time=true)
    println("Target explained variance: $opt")
    plt.figure()
    for nworkers in unique(df.nworkers)
        df_nworkers = df[df.nworkers .== nworkers, :]
        for nwait in unique(df_nworkers.nwait)
            if nwait < 6
                continue
            end
            df_nwait = df_nworkers[df_nworkers.nwait .== nwait, :]
            xs = zeros(0)
            ys = zeros(0)   
            ys .= -1         
            for nsubpartitions in unique(df_nwait.nsubpartitions)
                df_nsubpartitions = df_nwait[df_nwait.nsubpartitions .== nsubpartitions, :]   
                v = 0.0         
                for jobid in unique(df_nsubpartitions.jobid)
                    df_jobid = df_nsubpartitions[df_nsubpartitions.jobid .== jobid, :]
                    sort!(df_jobid, "iteration")
                    i = findfirst((v)->v >= atol+opt*(1-rtol), df_nsubpartitions.mse)
                    if isnothing(i)
                        continue
                    end
                    if time
                        v += df_jobid.t_total[i]
                    else
                        v += df_jobid.iteration[i]
                    end
                end
                if iszero(v)
                    v = -1
                end
                push!(xs, nsubpartitions)
                push!(ys, v)
            end
            plt.plot(xs, ys, "-o", label="($nworkers, $nwait)")
        end
    end    
    plt.grid()
    plt.legend()    
    plt.xlabel("nsubpartitions")
    if time
        plt.ylabel("convergence time")
    else    
        plt.ylabel("iterations until convergence")        
    end
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

function plot_linear_model(dct::AbstractDict; nworkers=unique(df.nworkers)[1])

    plt.figure()
    plt.title("Linear model fit iteration time (not counting decoding)")

    plt.subplot(3, 1, 1)    
    plt.ylabel("Time [s]")
    plt.grid()
    plt.title("Offset")    

    plt.subplot(3, 1, 2)
    plt.ylabel("Time [s]")
    plt.grid()
    plt.title("Time / flop")

    plt.subplot(3, 1, 3)
    plt.grid()
    plt.ylabel("Time [s]")
    plt.title("Time / communicated element")
    plt.xlabel("nwait")    

    for (label, df) in dct
        df = df[df.nworkers .== nworkers, :]

        nwait_all = unique(df.nwait)
        # df = copy(df)
        # df[:foo] = df.worker_flops .* df.communication
        models = [linear_model(Matrix(df[df.nwait .== nwait, [:worker_flops, :communication]]), df[df.nwait .== nwait, :t_compute]) for nwait in nwait_all]

        # offset
        plt.subplot(3, 1, 1)
        plt.plot(nwait_all, [model[1] for model in models], "o-", label=label)

        # worker flops slope
        plt.subplot(3, 1, 2)
        plt.plot(nwait_all, [model[2] for model in models], "o-", label=label)
        
        # communication slope
        plt.subplot(3, 1, 3)
        plt.plot(nwait_all, [model[3] for model in models], "o-", label=label)
    end

    plt.legend()
    return
end

plot_linear_model(df::AbstractDataFrame, args...; kwargs...) = plot_linear_model(Dict("df"=>df), args...; kwargs...)

"""

Plot iteration time against the number of elements processed
"""
function plot_iterationtime_flops(df)
    offsets = zeros(0)
    slopes = zeros(0)

    colors = Iterators.cycle(["r", "b", "g", "k", "m"])

    plt.figure()
    for (color, (nworkers, nwait, comm)) in zip(colors, Iterators.product(unique(df.nworkers), unique(df.nwait), unique(df.communication)))
        if nwait != 1
            continue
        end

        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        dfi = dfi[dfi.nwait .== nwait, :]
        dfi = dfi[dfi.communication .≈ comm, :]
        if size(dfi, 1) == 0
            continue
        end

        plt.plot(dfi.worker_flops, dfi.t_compute, "$(color).", label="Comm. $comm, $((nworkers, nwait))")
        gd = groupby(dfi, :worker_flops)
        ys = [mean(df.t_compute) for df in gd]
        xs = [mean(df.worker_flops) for df in gd]
        plt.plot(xs, ys, "$(color)s")

        # fit a line to the data
        nflops = dfi.worker_flops
        offset, slope = linear_model(dfi.worker_flops, dfi.t_compute)
        start, stop = 0, maximum(nflops)
        plt.plot(
            [start, stop],
            [start, stop].*slope .+ offset,
            "$(color)-"
        )

        # print the parameters            
        push!(offsets, offset)
        push!(slopes, slope)
        println("[nworkers: $nworkers, nwait: $nwait, communication: $comm] => offset: $offset, slope: $slope")
    end
    plt.xlabel("Flops")
    plt.ylabel("Iteration time [s]")
    plt.title("Time per iteration")
    plt.grid()
    plt.legend()
    # println("offsets: $offsets")
    # println("slopes: $slopes")
    return
end

"""

Plot the iteration time (compute) as a function of the amount of communication
"""
function plot_iterationtime_comm(df)
    offsets = zeros(0)
    slopes = zeros(0)

    plt.figure()
    for (nworkers, nwait, flops) in Iterators.product(unique(df.nworkers), unique(df.nwait), unique(df.worker_flops)[1:3])
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        dfi = dfi[dfi.nwait .== nwait, :]
        dfi = dfi[dfi.worker_flops .≈ flops, :]
        if size(dfi, 1) == 0
            continue
        end                    
        plt.plot(dfi.communication, dfi.t_compute, ".", label="($nworkers, $nwait, $flops)")

        # fit a line to the data
        offset, slope = linear_model(dfi.communication, dfi.t_compute)
        start, stop = 0, maximum(dfi.communication)
        plt.plot(
            [start, stop],
            [start, stop].*slope .+ offset,
        )

        # print the parameters            
        push!(offsets, offset)
        push!(slopes, slope)
        println("[nworkers: $nworkers, nwait: $nwait, nflops: $flops] => offset: $offset, slope: $slope")            
    end
    plt.xlabel("Communicated elements / iteration")
    plt.ylabel("Iteration time [s]")
    plt.title("Time per iteration")
    plt.grid()
    plt.legend()
    # println("offsets: $offsets")
    # println("slopes: $slopes")
    return
end

"""

Plot the best possible explained variance under Diggavi source data encoding.
"""
function plot_diggavi_convergence(coderates=range(0.1, 1, length=10))
    plt.figure()
    colors = ["b", "r", "k", "m"]
    for (nrows, ncols, ncomponents) in [(100, 20, 10), (1000, 200, 100), (1000, 200, 190), (1000, 200, 10)]

        # exact
        X = randn(nrows, ncols)        
        # X = test_matrix(nrows, ncols, ncomponents)
        V = pca(X, ncomponents)
        ev = explained_variance(X, V)
        color = pop!(colors)

        # Diggavi source data encoding
        evs = zeros(length(coderates))
        for (i, coderate) in enumerate(coderates)
            nc = ceil(Int, nrows/coderate)
            nc = 2^ceil(Int, log2(nc))
            GX = encode_hadamard(X, nc)
            GV = pca(GX, ncomponents)
            evs[i] = explained_variance(X, GV) / ev
        end
        plt.plot(coderates, evs, "$(color)o-", label="Diggavi $((nrows, ncols, ncomponents))")

        # # Systematic Diggavi encoding
        # for (i, coderate) in enumerate(coderates)
        #     nc = ceil(Int, nrows/coderate)
        #     GX = vcat(X, view(encode_hadamard(X, nc), 1:(nc-nrows), :))
        #     GV = pca(GX, ncomponents)
        #     evs[i] = explained_variance(X, GV) / ev
        # end
        # plt.plot(coderates, evs, "$(color)s-", label="Systematic Diggavi $((nrows, ncols, ncomponents))")        

        # # Unrelated random matrix
        # for (i, coderate) in enumerate(coderates)
        #     nc = ceil(Int, nrows/coderate)
        #     RX = randn(nc, ncols)
        #     RV = pca(RX, ncomponents)
        #     evs[i] = explained_variance(X, RV) / ev
        # end        
        # plt.plot(coderates, evs, "$(color)^-", label="Random $((nrows, ncols, ncomponents))")        
    end
    plt.legend()
    plt.grid()
    plt.ylim(0.9, 1.0)
    plt.xlabel("Code rate")
    plt.ylabel("Fraction retained explained variance (upper bound)")
    plt.title("Diggavi bound (randn matrix)")
end

function write_table(xs::AbstractVector, ys::AbstractVector, filename::AbstractString)
    length(xs) == length(ys) || throw(DimensionMismatch("xs has dimension $(length(xs)), but ys has dimension $(length(ys))"))
    open(filename, "w") do io
        for i in 1:length(xs)
            write(io, "$(xs[i]) $(ys[i])\n")
        end
    end
    return
end

function plot_genome_convergence(df, nworkers=unique(df.nworkers)[1], opt=maximum(df.mse))
    println("nworkers: $nworkers, opt: $opt")

    # Nn: 18
    
    # Np: 1 (Nw=16 best)
    ## Nw: 1
    # η: 0.2 to 0.6
    ## Nw: 3
    # η: 0.2
    ## Nw: 9
    # η: 0.1-0.9
    ## Nw: 16
    # η: 0.9    

    # Np: 2 (Nw=16 is best)
    ## Nw: 1
    # η: 0.3
    ## Nw: 3
    # η: 0.2
    ## Nw: 9
    # η: 0.6
    ## Nw: 16
    # η: 0.9

    # Np: 3 (Nw=16 is best)
    ## Nw: 1
    # η: 0.2, 0.7
    ## Nw: 3
    # η: 0.5
    ## Nw: 9
    # η: 0.3, 0.7
    ## Nw: 16
    # η: 0.9

    # Np: 4 (Nw=16 is best)
    ## Nw: 1
    # η: 0.4, 0.7
    ## Nw: 3
    # η: 0.7
    ## Nw: 9
    # η: 0.8
    ## Nw: 16
    # η: 0.8

    # Takeaway:
    # Using more partitions is better for the first few iterations
    # But at some point they cross, and using fewer partitions is better
    # (it makes the gradient more stable)
    # Having Nw < Nn gives no advantage whatsoever
    # It's better to just wait for all workers
    # Which is dispointing
    # I'd like to have a scenario where not waiting for all workers gives some advantage
    # In the data I have, there's no need for straggler mitigation
    # Which is actually what my data has been telling me
    # I know the point at which using more workers will slow you down
    # And I'm below that point
    # I've made it difficult for myself by having an iterate that's so enormous
    # And I don't have time to re-do the simulations for other datasets
    # So I need to do something clever
    # I could increase the amount of straggling just so that I get to a point where straggler mitigation makes sense
    # That'd be easy
    # Just say that there's no need for straggler mitigation, since we're below the limit
    # And then show what happens if you up the slope
    # At least that's what I should be doing for now
    # Since I need to wrap up this paper
    # So, first one plot showing that we don't need straggler mitigation
    # Then, another with increased slope

    # (nwait, nsubpartitions, stepsize)
    if nworkers == 6
        params = [
            (nworkers, 1, 1),  # full GD
            (nworkers, 10, 0.9), # sub-partitioning            
            (1, 1, 0.9), # straggler mitigation
            # (1, 10, 0.9),
            (2, 10, 0.9), # straggler mitigation + sub-partititoning
            # (4, 10, 0.9),     
        ]        
    elseif nworkers == 12
        params = [
            (nworkers, 1, 1), 
            (1, 1, 0.9),
            (nworkers, 5, 0.9),
            (1, 5, 0.9),
        ]
    elseif nworkers == 18
        Nw = 16
        Np = 4
        params = [
            (nworkers, 1, 1.0), # full GD
            (nworkers, 4, 0.9), # sub-partitioning
            (1, 1, 0.2), # straggler mitigation
            (3, 4, 0.7),
            # (nworkers, 3, 0.9),
            # (nworkers, 4, 0.9),
            # (16, 1, 0.9),
            # (16, 2, 0.9),
            # (16, 3, 0.9),
            # (16, 4, 0.8),
            # (9, 4, 0.4),
            # (Nw, Np, 0.1),
            # (Nw, Np, 0.2),
            # (Nw, Np, 0.3),
            # (Nw, Np, 0.4),
            # (Nw, Np, 0.5),
            # (Nw, Np, 0.6),            
            # (Nw, Np, 0.7),
            # (Nw, Np, 0.8),
            # (Nw, Np, 0.9),
            # (nworkers, 4, 0.9),
            # (1, 4, 0.9),
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

    # plot the bound
    r = 2
    Nw = nworkers * (1/6)

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
    x0 = get_offset(worker_flops) .+ get_slope(worker_flops, nworkers) * Nw
    xs = x0 .* (1:maximum(dfi.iteration))

    # make the plot
    plt.figure()        
    plt.semilogy(xs, ys, "--k", label="Bound r: $r, Nw: $Nw")
    write_table(xs, ys, "./data/bound_$(nworkers)_$(Nw)_$(r).csv")
    # plt.semilogy([x0, x0], [1, 0], "--k", label="Bound r: $r, Nw: $Nw")

    # What I want to plot
    # Let's have two plots for the regular data and two for more straggling
    # For each scenario, 1 plot for 6 workers and 1 for 18 workers
    # Since I want to show the spread
    # I need to carefully choose which lines to plot
    # Regular batch GD
    # Waiting for all workers, but with sub-partitioning
    # Waiting for a subset of the workers, no sub-partitioning
    # Waiting for a subset of the workers, with sub-partitioning
    # DSAG, SAG, SGD, 

    for (nwait, nsubpartitions, stepsize) in params
        dfi = df
        dfi = dfi[dfi.nwait .== nwait, :]
        dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]    
        dfi = dfi[dfi.stepsize .== stepsize, :]
        println("nwait: $nwait, nsubpartitions: $nsubpartitions, stepsize: $stepsize")

        ### DSAG
        dfj = dfi
        dfj = dfj[dfj.variancereduced .== true, :]
        dfj = dfj[dfj.nostale .== false, :]
        filename = "./data/dsag_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"
        println("SAG: $(length(unique(dfj.jobid))) jobs")
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
        dfj = dfj[dfj.nostale .== true, :]
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

    if nworkers == 6
        plt.xlim(0, 120)
        # plt.xlim(0, 240)
        plt.ylim(1e-5, 1e-1) 
    elseif nworkers == 18
        plt.xlim(0, 60)
        plt.ylim(1e-5, 1e-1)         
    else
        plt.xlim(0, 80)
        plt.ylim(1e-7, 1e-1)        
    end
    plt.xlabel("Time [s]")
    plt.ylabel("Explained Variance Sub-optimality Gap")

    # tikzplotlib.save("./plots/genome_convergence_2.0.tex")
    return
end