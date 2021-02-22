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

Read a csv file into a DataFrame
"""
function read_df(directory="C:/Users/albin/Dropbox/Eigenvector project/data/dataframes/pca/210208/")
    filename = sort!(glob("*.csv", directory))[end]
    println("Reading $filename")
    df = DataFrame(CSV.File(filename, normalizenames=true))
    df[:nostale] = Missings.replace(df.nostale, false)
    df[:kickstart] = Missings.replace(df.kickstart, false)
    df = df[.!ismissing.(df.nworkers), :]
    df = df[df.kickstart .== false, :]
    df = remove_initialization_delay!(df)
    df[:worker_flops] = worker_flops_from_df(df)
    df.npartitions = df.nworkers .* df.nsubpartitions
    rename!(df, :t_compute => :latency)

    # scale up workload
    # df[:worker_flops] .*= 22
    # df[:t_compute] .= model_tcompute_from_df(df, samp=1)

    df[:nbytes] = df.ncolumns .* df.ncomponents .* 8
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
            threshold = 0.01 # for interval of 10 s
        elseif isapprox(worker_flops, 2.52e7, rtol=1e-2)
            threshold = 0.005 # for interval of 10 s
        end        
        windowsize = ceil(Int, intervalsize/(maximum(x.time)/maximum(x.iteration)))
        vs = runmean(float.(x.worker_compute_latency), windowsize) .- minimum(x.worker_compute_latency)
        burst = vs .>= threshold

        # also mark samples in half a windowsize before each window
        for i in findall(isone, diff(burst))
            j = max(1, i-round(Int, windowsize/2))
            burst[j:i] .= true
        end
        burst
    end
    sort!(dfo, [:jobid, :worker_index, :iteration])    
    by(dfo, [:jobid, :worker_index], [:worker_compute_latency, :time, :iteration, :worker_flops] => f => :burst).burst
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

Read a latency experiment csv file into a DataFrame
"""
function read_latency_df(directory="C:/Users/albin/Dropbox/Eigenvector project/data/dataframes/latency/210215_v3/")
    filename = sort!(glob("*.csv", directory))[end]
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