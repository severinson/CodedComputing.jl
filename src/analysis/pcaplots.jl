using CSV, DataFrames, PyPlot, Statistics

using PyCall
tikzplotlib = pyimport("tikzplotlib")

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

Set t_compute to globally averaged values to minimize the effect of between-run variations.
These values are only valid for 1000 genomes chromosome 20 results.
Set `samp` to a value larger than 1 to increase the effect of straggling, and to a value in [0, 1) to reduce the effect.
"""
function genome_cleanup_tcompute!(df; samp=1.0)

    # There are 4 variations of t_compute
    # I also need to handle kickstart

    # npartitions: 1
    offset, slope = 10.64949, 0.39459 * samp
    mask = (df.nsubpartitions .== 1) .& (df.nreplicas .== 1)
    df[mask, "t_compute"] .= offset .+ slope.*df[mask, "nwait"]
    mask .&= df.kickstart .== true .& df.iteration .== 1
    df[mask, "t_compute"] .= offset .+ slope.*df[mask, "nworkers"]

    # npartitions: 2
    offset, slope = 4.89508, 0.19334 * samp
    mask = (df.nsubpartitions .== 2) .& (df.nreplicas .== 1)
    df[mask, "t_compute"] .= offset .+ slope.*df[mask, "nwait"]
    mask .&= df.kickstart .== true .& df.iteration .== 1
    df[mask, "t_compute"] .= offset .+ slope.*df[mask, "nworkers"]    

    # npartitions: 3
    offset, slope = 3.25924, 0.12786 * samp
    mask = (df.nsubpartitions .== 3) .& (df.nreplicas .== 1)
    df[mask, "t_compute"] .= offset .+ slope.*df[mask, "nwait"]   
    mask .&= df.kickstart .== true .& df.iteration .== 1
    df[mask, "t_compute"] .= offset .+ slope.*df[mask, "nworkers"]      

    # npartitions: 5
    offset, slope = 2.03563, 0.08969 * samp
    mask = (df.nsubpartitions .== 5) .& (df.nreplicas .== 1)
    df[mask, "t_compute"] .= offset .+ slope.*df[mask, "nwait"]
    mask .&= df.kickstart .== true .& df.iteration .== 1
    df[mask, "t_compute"] .= offset .+ slope.*df[mask, "nworkers"]    

    # [nreplicas: 1, nflops: 1.16158821e8] offset: 10.64949 (9.168043193980956e-8), slope: 0.39459 (3.3969998701485833e-9)
    # [nreplicas: 1, nflops: 5.80794105e7] offset: 4.89508 (8.428249381362707e-8), slope: 0.19334 (3.3288744194546834e-9)    

    # [nreplicas: 1, nflops: 3.8719607e7] offset: 3.25924 (8.417531955098812e-8), slope: 0.12786 (3.302232136648561e-9)    
    # [nreplicas: 1, nflops: 2.32317642e7] offset: 2.03563 (8.762258491098293e-8), slope: 0.08969 (3.860868172267541e-9)
    df
end

"""

Return a vector composed of the cumulative compute time for each job.
"""
function cumulative_time_from_df(df)
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

function read_df(filename="data/pca/1000genomes/aws12/210114_v5.csv")
    df = DataFrame(CSV.File(filename, normalizenames=true))
    df[:kickstart] = Missings.replace(df.kickstart, false)
    df = remove_initialization_delay!(df)    
    df = genome_cleanup_tcompute!(df, samp=2.0) # only for 1000 genomes chromosome 20
    df[:t_total] = cumulative_time_from_df(df)
    df[:worker_flops] = worker_flops_from_df(df)
    df[:communication] = communication_from_df(df)
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
        for (nreplicas, worker_flops) in Iterators.product(unique(df.nreplicas), unique(df.worker_flops))
            dfi = df
            dfi = dfi[dfi.nreplicas .== nreplicas, :]
            dfi = dfi[dfi.worker_flops .== worker_flops, :]
            # dfi = dfi[dfi.kickstart .!= true, :]
            dfi = dfi[Missings.replace(dfi.kickstart, false) .== false, :]
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
            l = label * " nrep=$nreplicas, nflops=$(round(worker_flops, sigdigits=3))"            
            yerr = zeros(2, length(xs))
            yerr[1, :] .= ys .- mins
            yerr[2, :] .= maxes .- ys
            plt.errorbar(xs, ys, yerr=yerr, fmt=".", label=l)

            # plot a linear model fit to the data
            offset, slope = linear_model(xs, ys)
            plt.plot([0, maximum(xs)], offset .+ [0, maximum(xs)*slope])
            @assert length(unique(dfi.nrows)) == 1
            nrows = unique(dfi.nrows)[1]

            println("[nreplicas: $nreplicas, nflops: $worker_flops] offset: $(round(offset, digits=5)) ($(offset / worker_flops)), slope: $(round(slope, digits=5)) ($(slope / worker_flops))")
        end
    end

    plt.grid()
    plt.legend()
    plt.xlabel("nwait")
    plt.ylabel("Compute time [s]")

    # tikzplotlib.save("./plots/tcompute.tex")
    
    return
end

plot_iterationtime_quantiles(df::AbstractDataFrame) = plot_iterationtime_quantiles(Dict("df"=>df))

"""

Plot the update time at the master against the number of sub-partitions.
"""
function plot_updatetime(dct)

    plt.figure()
    for (label, df) in dct

        for (nreplicas, nsubpartitions) in Iterators.product(unique(df.nreplicas), unique(df.nsubpartitions))

            if nreplicas != 1
                continue
            end
            # if nsubpartitions != 1
            #     continue
            # end

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

plot_updatetime(df::AbstractDataFrame) = plot_updatetime(Dict("df"=>df))

"""

Plot the time per iteration against the iteration index.
"""
function plot_iterationtime_traces(dct)
    plt.figure()
    for (label, df) in dct
        for (nreplicas, nwait) in Iterators.product(unique(df.nreplicas), unique(df.nwait))
            dfi = df
            dfi = dfi[dfi.nreplicas .== nreplicas, :]
            dfi = dfi[dfi.nwait .== nwait, :]
            if size(dfi, 1) == 0
                continue
            end
            l = label * " nrep=$nreplicas, nwait=$nwait" 
            plt.plot(dfi.iteration, dfi.t_compute, ".", label=l)
        end
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
        println("Best for t <= $(t): $(df_t.jobid[j]) (nsubpartitions = $(df_t.nsubpartitions[j]), nwait = $(df_t.nwait[j]))")
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

plot_timeseries_best(df::AbstractDataFrame) = plot_timeseries_best(Dict("df"=>df))

"""

Plot the MSE as a function of time (or iteration) separately for each unique job.
"""
function plot_timeseries(df; time=true, filters=Dict{String,Any}(), prune=false, opt=nothing)
    df = dropmissing(df)
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

function plot_iterationtime_combined(df)
    colors = Iterators.cycle(["r", "b", "g", "k", "m"])
    # plt.figure()
    fig = plt.figure()
    ax = fig[:add_subplot](111, projection="3d")    
    for (color, (nworkers, nwait)) in zip(colors, Iterators.product(unique(df.nworkers), unique(df.nwait)))
        if !(nwait in [1, 24, 47])
            continue
        end

        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        dfi = dfi[dfi.nwait .== nwait, :]
        if size(dfi, 1) == 0
            continue
        end

        gd = groupby(dfi, [:worker_flops, :communication])
        xs = [mean(df.worker_flops) for df in gd]
        ys = [mean(df.communication) for df in gd]
        zs = [mean(df.t_compute) for df in gd]
        ax[:plot](xs, ys, zs, "o", label="$((nworkers, nwait))")

        # ax[:plot](dfi.worker_flops, dfi.communication, dfi.t_compute, ".", label="$((nworkers, nwait))")
    end
    # ax[:set_xlabel]("Flops")
    plt.xlabel("Flops")
    plt.ylabel("Communication")
    ax[:set_zlabel]("Iteration time [s]")
    # plt.zlabel("Iteration time [s]")
    plt.title("Time per iteration")
    plt.grid()
    plt.legend()
    # println("offsets: $offsets")
    # println("slopes: $slopes")
    return
end

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

function foo(df)
    plt.figure()

    # SAG
    for jobid in [2, 10, 17]
        dfi = df[df.jobid .== jobid, :]
        t_compute = mean(dfi.t_compute)
        t_update = mean(dfi.t_update)
        println("[SAG nwait=1] compute:\t$t_compute, update: $t_update")
        plt.plot(dfi.t_total, dfi.mse, "-o", label="VR, nwait=1, jobid=$jobid (ours)")
    end

    dfi = df[df.jobid .== 19, :]
    t_compute = mean(dfi.t_compute)
    t_update = mean(dfi.t_update)
    println("[SAG nwait=2] compute:\t$t_compute, update: $t_update")
    plt.plot(dfi.t_total, dfi.mse, "-o", label="VR, nwait=2 (ours)")

    dfi = df[df.jobid .== 4, :]
    t_compute = mean(dfi.t_compute)
    t_update = mean(dfi.t_update)
    println("[SAG nwait=10] compute:\t$t_compute, update: $t_update")
    plt.plot(dfi.t_total, dfi.mse, "-o", label="VR, nwait=10 (ours)")

    dfi = df[df.jobid .== 6, :]
    t_compute = mean(dfi.t_compute)
    t_update = mean(dfi.t_update)
    println("[SAG nwait=11] compute:\t$t_compute, update: $t_update")    
    plt.plot(dfi.t_total, dfi.mse, "-o", label="VR, nwait=11 (ours)")

    dfi = df[df.jobid .== 8, :]
    t_compute = mean(dfi.t_compute)
    t_update = mean(dfi.t_update)
    println("[SAG nwait=12] compute:\t$t_compute, update: $t_update")    
    plt.plot(dfi.t_total, dfi.mse, "-o", label="VR, nwait=12 (ours)")

    # # SGD
    # dfi = df[df.jobid .== 3, :]
    # t_compute = mean(dfi.t_compute)
    # t_update = mean(dfi.t_update)
    # println("[SGD nwait=10] compute:\t$t_compute, update: $t_update")        
    # plt.plot(dfi.t_total, dfi.mse, "-s", label="SGD, nwait=10")

    # dfi = df[df.jobid .== 5, :]
    # t_compute = mean(dfi.t_compute)
    # t_update = mean(dfi.t_update)
    # println("[SGD nwait=11] compute:\t$t_compute, update: $t_update")        
    # plt.plot(dfi.t_total, dfi.mse, "-s", label="SGD, nwait=11")

    # dfi = df[df.jobid .== 7, :]
    # t_compute = mean(dfi.t_compute)
    # t_update = mean(dfi.t_update)
    # println("[SGD nwait=12] compute:\t$t_compute, update: $t_update")        
    # plt.plot(dfi.t_total, dfi.mse, "-s", label="SGD, nwait=12") 

    plt.xlabel("Time [s]")
    plt.ylabel("Explained variance")
    plt.grid()
    plt.title("AWS, 12 workers, genomics dataset, 3 components")
    plt.legend()
    # plt.ylim(0.6351, 0.6355)
    # plt.xlim(20, 90)
    return
end

function plot_genome_convergence(df)

    opt = maximum(df.mse)
    println("opt: $opt")

    plt.figure()

    ## nwait=12, nsubpartitions=1
    dfi = df
    dfi = dfi[dfi.nreplicas .== 1, :]
    dfi = dfi[dfi.nsubpartitions .== 1, :]    
    dfi = dfi[dfi.nwait .== 12, :]
    dfi = dfi[dfi.stepsize .== 1.0, :]
    dfi = dfi[Missings.replace(dfi.kickstart, false) .== false, :]

    ### SAG
    dfj = dfi[dfi.variancereduced .== true, :]
    dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)
    plt.semilogy(dfj.t_total_mean, opt.-dfj.mse_mean, "o-", label="SAG (nwait: 12, nsubpartitions: 1)")

    ### SGD    
    dfj = dfi[dfi.variancereduced .== false, :]
    dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)
    plt.semilogy(dfj.t_total_mean, opt.-dfj.mse_mean, "s-", label="SGD (nwait: 12, nsubpartitions: 1)")

    ## nwait=1, nsubpartitions=1
    dfi = df
    dfi = dfi[dfi.nreplicas .== 1, :]
    dfi = dfi[dfi.nsubpartitions .== 1, :]    
    dfi = dfi[dfi.nwait .== 1, :]
    dfi = dfi[Missings.replace(dfi.kickstart, false) .== false, :]

    ### SAG
    dfj = dfi
    dfj = dfj[dfj.variancereduced .== true, :]
    dfj = dfj[dfj.stepsize .== 0.9, :]
    dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)
    plt.semilogy(dfj.t_total_mean, opt.-dfj.mse_mean, "o-", label="SAG (nwait: 1, nsubpartitions: 1)")

    ### SGD    
    dfj = dfi
    dfj = dfi[dfj.variancereduced .== false, :]
    dfj = dfj[dfj.stepsize .== 0.9, :]
    dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)
    plt.semilogy(dfj.t_total_mean, opt.-dfj.mse_mean, "s-", label="SGD (nwait: 1, nsubpartitions: 1)")

    ## nwait=12, nsubpartitions=5
    dfi = df
    dfi = dfi[dfi.nreplicas .== 1, :]
    dfi = dfi[dfi.nsubpartitions .== 5, :]    
    dfi = dfi[dfi.nwait .== 12, :]
    dfi = dfi[Missings.replace(dfi.kickstart, false) .== false, :]

    ### SAG
    dfj = dfi
    dfj = dfj[dfj.variancereduced .== true, :]
    dfj = dfj[dfj.stepsize .== 0.9, :]
    # for jobid in unique(dfj.jobid)
    #     dfk = dfj
    #     dfk = dfk[dfk.jobid .== jobid, :]
    #     plt.semilogy(dfk.t_total, opt.-dfk.mse, ".-", label="SAG (nwait: 12, nsubpartitions: 5)")
    # end    
    dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)
    plt.semilogy(dfj.t_total_mean, opt.-dfj.mse_mean, "o-", label="SAG (nwait: 12, nsubpartitions: 5)")

    ### SGD    
    dfj = dfi
    dfj = dfi[dfj.variancereduced .== false, :]
    dfj = dfj[dfj.stepsize .== 0.9, :]
    # for jobid in unique(dfj.jobid)
    #     dfk = dfj
    #     dfk = dfk[dfk.jobid .== jobid, :]
    #     plt.semilogy(dfk.t_total, opt.-dfk.mse, ".-", label="SGD (nwait: 12, nsubpartitions: 5)")        
    # end        
    dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)
    plt.semilogy(dfj.t_total_mean, opt.-dfj.mse_mean, "s-", label="SGD (nwait: 12, nsubpartitions: 5)")      

    ## nwait=1, nsubpartitions=5
    dfi = df
    dfi = dfi[dfi.nreplicas .== 1, :]
    dfi = dfi[dfi.nsubpartitions .== 5, :]    
    dfi = dfi[dfi.nwait .== 1, :]
    dfi = dfi[Missings.replace(dfi.kickstart, false) .== false, :]

    ### SAG
    dfj = dfi
    dfj = dfj[dfj.variancereduced .== true, :]
    dfj = dfj[dfj.stepsize .== 0.9, :]
    # for jobid in unique(dfj.jobid)
    #     dfk = dfj
    #     dfk = dfk[dfk.jobid .== jobid, :]
    #     plt.semilogy(dfk.t_total, opt.-dfk.mse, ".-", label="SAG (nwait: 1, nsubpartitions: 5)")        
    # end    
    dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)
    plt.semilogy(dfj.t_total_mean, opt.-dfj.mse_mean, "o-", label="SAG (nwait: 1, nsubpartitions: 5)")

    ### SGD    
    dfj = dfi
    dfj = dfi[dfj.variancereduced .== false, :]
    dfj = dfj[dfj.stepsize .== 0.9, :]
    # for jobid in unique(dfj.jobid)
    #     dfk = dfj
    #     dfk = dfk[dfk.jobid .== jobid, :]
    #     plt.semilogy(dfk.t_total, opt.-dfk.mse, ".-", label="SAG (nwait: 1, nsubpartitions: 5)")        
    # end        
    dfj = combine(groupby(dfj, :iteration), :mse => mean, :t_total => mean)
    plt.semilogy(dfj.t_total_mean, opt.-dfj.mse_mean, "s-", label="SGD (nwait: 1, nsubpartitions: 5)")        

    plt.grid()
    plt.legend()
    plt.ylim(1e-7, 1e-1)
    plt.xlim(0, 80)
    plt.xlabel("Time [s]")
    plt.ylabel("Explained Variance Sub-optimality Gap")

    tikzplotlib.save("./plots/genome_convergence_2.0.tex")
    return
end