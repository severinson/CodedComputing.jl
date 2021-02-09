# Code for loading .csv files into a DataFrame and for pre-processing that data

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

"""

Read a csv file into a DataFrame
"""
function read_df(filename="C:/Users/albin/Dropbox/Eigenvector project/data/dataframes/210208/210208_v5.csv"; nworkers=nothing)
    df = DataFrame(CSV.File(filename, normalizenames=true))
    df[:nostale] = Missings.replace(df.nostale, false)
    df[:kickstart] = Missings.replace(df.kickstart, false)
    df = df[.!ismissing.(df.nworkers), :]
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
    df
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