using CSV, DataFrames, PyPlot, Statistics

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


function read_df(filename="data/pca/10000_5000_100_4.csv")    
    df = DataFrame(CSV.File(filename, normalizenames=true))
    df[:t_total] = cumulative_time_from_df(df)
    df
end

"""

Plot the CCDF of the iteration time for all values of `nwait` for the given number of workers.
"""
function plot_iterationtime_cdf(df; nworkers::Integer=9)
    df = df[df.iteration .>= 2000, :]
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
function plot_iterationtime_quantiles(df; nworkers::Integer=9, qs=[0.5, 0.9, 0.99])
    df = df[df.iteration .>= 10, :]
    df = df[df.nworkers .== nworkers, :]
    x = 1:nworkers
    plt.figure()
    for q in qs
        y = [quantile(df[df.nwait .== nwait, :].t_compute, q) for nwait in x]
        plt.plot(x, y, label="$(q)-th quantile")
    end
    plt.grid()
    plt.legend()
    plt.xlim(1, nworkers)
    plt.xlabel("nwait")
    plt.ylabel("Iteration time [s]")
end

"""

Plot the time per iteration against the iteration index.
"""
function plot_iterationtime_traces(df; nworkers::Integer=9, q::Real=0.5, nwaits=1:nworkers)
    df = df[df.nworkers .== nworkers, :]
    # df = df[df.algorithm .== "pcacsc.jl", :]
    plt.figure()
    niterations = 0
    for nwait in nwaits
        df_nwait = df[df.nwait .== nwait, :]
        niterations = max(niterations, size(df_nwait, 1))
        x = df_nwait.iteration
        # x = cumsum(df_nwait.t_compute .+ df_nwait.t_update)
        plt.plot(x, df_nwait.t_compute, ".", label="nwait=$nwait")
    end
    plt.xlim(0, niterations)
    plt.legend()
    plt.grid()
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
function plot_convergence_rate(df, iteration::Integer=20; filters=Dict{String,Any}())
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

"""

Plot the MSE as a function of time (or iteration) separately for each unique job.
"""
function plot_timeseries(df; time=true, filters=Dict{String,Any}())
    df = dropmissing(df)
    for (key, value) in filters
        df = df[df[key] .== value, :]
    end

    # TODO: temporary
    df = df[df.variancereduced .== 1, :]
    markers = Dict{Int,String}(1 => "o-", 2 => "s-", 5 => "^-", 10 => "d-")

    plt.figure()
    for jobid in unique(df.jobid)
        df_jobid = df[df.jobid .== jobid, :]
        nwait = df_jobid.nwait[1]
        
        # TODO: temporary
        v1 = unique(df_jobid.nsubpartitions)[1]
        v2 = unique(df_jobid.stepsize)[1]
        label = "job $jobid ($v1, $v2)"
        marker = markers[v1]

        # plot convergence
        if time
            plt.plot(df_jobid.t_total, df_jobid.mse, marker, label=label)
        else
            x = df_jobid.iteration
            plt.plot(x, df_jobid.mse, marker, label="job $jobid ($pfraction, $nsubpartitions)")            
        end
    end
    plt.xlabel("Time [s]")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()
end

"""

Plot the time (or number of iterations) until the explained variance has converged to within
`atol + rtol*opt` of `opt` as a function of `nsubpartitions` separately for each unique 
(`nworkers`, `nwait`) pair. Use this plot to select optimal values for `nwait` and `nsubpartitions` 
for a given number of workers.
"""
function plot_convergence_time(df; opt=maximum(df.mse), atol=0, rtol=1e-4, time=true)
    plt.figure()
    for nworkers in unique(df.nworkers)
        df_nworkers = df[df.nworkers .== nworkers, :]
        for nwait in unique(df_nworkers.nwait)
            # if nwait < 6
            #     continue
            # end
            df_nwait = df_nworkers[df_nworkers.nwait .== nwait, :]
            xs = zeros(0)
            ys = zeros(0)            
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

Plot iteration time against the number of elements processed
"""
function plot_iterationtime_elements(df, nrows=unique(df.nrows)[1], ncolumns=unique(df.ncolumns)[1], ncomponents=unique(df.ncomponents)[1])
    df = df[df.nrows .== nrows, :]
    df = df[df.ncolumns .== ncolumns, :]
    df = df[df.ncomponents .== ncomponents, :]
    for nworkers in unique(df.nworkers)
        df_nworkers = df[df.nworkers .== nworkers, :]
        for nwait in unique(df_nworkers.nwait)
            df_nwait = df_nworkers[df_nworkers.nwait .== nwait, :]
            nelements = float.(df_nwait.nrows .* df_nwait.ncolumns)
            nelements .*= df_nwait.pfraction
            nelements ./= df_nwait.nsubpartitions
            plt.plot(nelements, df_nwait.t_compute, ".", label="($nworkers, $nwait)")

            # fit a line to the data
            A = ones(length(nelements), 2)
            A[:, 2] .= nelements
            offset, slope = A \ df_nwait.t_compute
            start, stop = 0, maximum(nelements)
            plt.plot(
                [start, stop],
                [start, stop].*slope .+ offset,
            )

            # print the parameters
            println("[nworkers: $nworkers, nwait: $nwait] => offset: $offset, slope: $slope")
        end
    end
    plt.xlabel("Number of elements")
    plt.ylabel("Iteration time [s]")
    plt.title("Time per iteration for (n, m, k) = $((nrows, ncolumns, ncomponents))")
    plt.grid()
    plt.legend()
    return
end