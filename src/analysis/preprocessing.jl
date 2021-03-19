# Code for loading .csv files into a DataFrame and for pre-processing that data
using Glob

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
        rv[mask] .= cumsum(df_jobid.latency .+ df_jobid.t_update)
    end
    rv
end

"""

Return a vector composed of the number of flops performed by each worker and iteration.

"""
function worker_flops_from_df(df; density=0.05360388070027386)
    nflops = float.(df.nrows)
    nflops ./= df.nworkers
    nflops .*= df.nreplicas
    nflops ./= Missings.replace(df.nsubpartitions, 1.0)
    nflops .*= 2 .* df.ncolumns .* df.ncomponents
    nflops .*= density
end

"""

Read a csv file into a DataFrame
"C:/Users/albin/Dropbox/Eigenvector project/dataframes/pca/sprand/210312/"
"""
function read_df(directory="C:/Users/albin/Dropbox/Eigenvector project/dataframes/pca/1000genomes_shuffled/210316/")
    filename = sort!(glob("*.csv", directory))[end]
    println("Reading $filename")
    df = DataFrame(CSV.File(filename, normalizenames=true))
    df = df[.!ismissing.(df.nworkers), :]
    df = df[.!ismissing.(df.iteration), :]
    df[:nostale] = Missings.replace(df.nostale, false)
    df[:kickstart] = Missings.replace(df.kickstart, false)
    df = df[df.kickstart .== false, :]
    df = remove_initialization_delay!(df)
    df[:worker_flops] = worker_flops_from_df(df)
    df.npartitions = df.nworkers .* df.nsubpartitions
    rename!(df, :t_compute => :latency)
    df[:nbytes] = df.nrows .* df.ncomponents .* 4
    df[:t_total] = cumulative_time_from_df(df)    
    df
end

"""

Return a boolean vector indicating for which samples the worker is currently experiencing a latency burst
"""
function burst_state_from_orderstats_df(dfo; intervalsize=5)
    function f(x)
        threshold = Inf
        worker_flops = mean(x.worker_flops)
        if isapprox(worker_flops, 7.56e7, rtol=1e-2)
            threshold = 0.01 # for 5s window
        elseif isapprox(worker_flops, 2.52e7, rtol=1e-2)
            threshold = 0.0025 # for 5s window
        elseif isapprox(worker_flops, 1.51e8, rtol=1e-2)
            threshold = 0.02
        elseif isapprox(worker_flops, 5.04e7, rtol=1e-2)
            threshold = 0.005
        elseif isapprox(worker_flops, 3.78e7, rtol=1e-2)
            threshold = 0.004
        elseif isapprox(worker_flops, 3.02e7, rtol=1e-2)
            threshold = 0.0035
        elseif isapprox(worker_flops, 1.51e7, rtol=1e-2)
            threshold = 0.001
        end        
        windowsize = ceil(Int, intervalsize/(maximum(x.time)/maximum(x.iteration)))
        # vs = runmean(float.(x.worker_compute_latency), windowsize) .- minimum(x.worker_compute_latency)
        vs = runmean(float.(x.worker_compute_latency), windowsize)
        vs .-= minimum(vs)
        burst = vs .>= threshold

        # shift the mean left by half a window size
        burst .= circshift(burst, -round(Int, windowsize/2))

        burst
    end
    sort!(dfo, [:jobid, :worker_index, :iteration])    
    by(dfo, [:jobid, :worker_index], [:worker_compute_latency, :time, :iteration, :worker_flops] => f => :burst).burst
end

"""

Return a df composed of the order statistic samples for each worker, iteration, and job.
"""
function orderstats_df(df; extend=false)
    df = df[.!ismissing.(df["latency_worker_1"]), :]    
    if size(df, 1) == 0
        return DataFrame()
    end    

    # determine the number of workers based on column names
    # (since not all rows may be present)
    cols = [name for name in names(df) if occursin("repoch", name)]
    indices = parse.(Int, last.(split.(cols, "_")))
    nworkers = maximum(indices)
    
    ### stack by worker latency
    df1 = select(df, Not(["repoch_worker_$i" for i in 1:nworkers]))
    if "compute_latency_worker_1" in names(df)
        select!(df1, Not(["compute_latency_worker_$i" for i in 1:nworkers]))    
    end    
    df1 = stack(df1, ["latency_worker_$i" for i in 1:nworkers], value_name=:worker_latency)
    select!(df1, Not(:variable), :variable => ((x) -> parse.(Int, last.(split.(x, "_")))) => :worker_index)
    dropmissing!(df1, :worker_latency)
    
    # stack by worker receive epoch
    df2 = stack(df, ["repoch_worker_$i" for i in 1:nworkers], [:jobid, :iteration], value_name=:repoch)
    select!(df2, Not(:variable), :variable => ((x) -> parse.(Int, last.(split.(x, "_")))) => :worker_index)    
    dropmissing!(df2, :repoch)
    df2[:isstraggler] = df2.repoch .< df2.iteration
    joined = innerjoin(df1, df2, on=[:jobid, :iteration, :worker_index]) 

    # stack by worker compute latency
    if "compute_latency_worker_1" in names(df)
        df3 = stack(df, ["compute_latency_worker_$i" for i in 1:nworkers], [:jobid, :iteration], value_name=:worker_compute_latency)
        select!(df3, Not(:variable), :variable => ((x) -> parse.(Int, last.(split.(x, "_")))) => :worker_index)            
        dropmissing!(df3, :worker_compute_latency)
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

    if extend
        dfi = joined[joined.nwait .== joined.nworkers, :]
        dfi.mse = missing
        for i in 1:maximum(df.nworkers)
            dfj = dfi
            dfj = dfj[dfj.nworkers .> i, :]        
            dfj = dfj[dfj.worker_index .<= i, :]
            if size(dfj, 1) == 0
                continue
            end
            dfj.nworkers .= i
            dfj.nwait .= i
            sort!(dfj, [:jobid, :iteration, :worker_latency])
            dfj.order .= by(dfj, [:jobid, :iteration], :nworkers => ((x) -> collect(1:maximum(x))) => :order).order
            joined = vcat(joined, dfj)
        end
    end
    # add a flag indicating if the worker is experiencing a latency burst
    # joined.burst = burst_state_from_orderstats_df(joined)

    return joined
end

"""

Compute the running mean of worker compute latency over windows of length `windowlengths` seconds.
"""
function compute_rmeans(dfo; miniterations=10000, windowlengths=[100, 10, 0.1, 0.01])
    dfo = dfo[dfo.niterations .>= miniterations, :]    
    sort!(dfo, [:jobid, :worker_index, :iteration])
    function f(x)
        vs = zeros(length(x.worker_compute_latency))
        rv = zeros(length(x.worker_compute_latency), length(windowlengths)+1)
        rv[:, 1] .= x.iteration
        for (i, windowlength) in enumerate(windowlengths)
            if iszero(windowlength)
                rv[:, i+1] .= float.(x.worker_compute_latency)
            elseif isinf(windowlength)
                rv[:, i+1] .= mean(x.worker_compute_latency)
            else
                windowsize = ceil(Int, windowlength/(maximum(x.time)/maximum(x.iteration)))
                # weights = DSP.Windows.gaussian(windowsize, 10)
                # weights ./= sum(weights)
                # rv[:, i+1] .= runmean(float.(x.worker_compute_latency), windowsize, weights)
                circshift!(view(rv, :, i+1), runmean(float.(x.worker_compute_latency), windowsize), -round(Int, windowsize/2))                
            end
            rv[:, i+1] .-= vs
            vs .+= rv[:, i+1]
        end
        rv
    end
    df = by(dfo, [:jobid, :worker_index], [:worker_compute_latency, :time, :iteration, :worker_flops] => f)    

    # fix the iteration type and name
    df.iteration = Int.(df.x1) 
    select!(df, Not(:x1))
    
    # rename running mean columns
    for (i, windowlength) in enumerate(windowlengths)    
        rename!(df, "x$(i+1)" => "rmean_$windowlength")
    end
    innerjoin(dfo, df, on=[:jobid, :worker_index, :iteration])
end

"""

Read a latency experiment csv file into a DataFrame
"""
function read_latency_df(directory="C:/Users/albin/Dropbox/Eigenvector project/dataframes/latency/210215_v3/")
    # filename = sort!(glob("*.csv", directory))[end]
    filename = directory*"df_v12.csv"
    println("Reading $filename")
    df = DataFrame(CSV.File(filename, normalizenames=true))
    df.worker_flops = 2 .* df.nrows .* df.ncols .* df.ncomponents .* df.density
    sort!(df, [:jobid, :iteration])
    df.time = by(df, :jobid, :latency => cumsum => :time).time # cumulative time since the start of the computation
    df[df.ncols .== 1812842, :], df[df.ncols .== 2504, :]
end

"""

Split the DataFrame by algorithm (sgd, dsag, sag, gd)
"""
function split_df_by_algorithm(df)
    error("Not implemented")
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