# Code for parsing the .h5 files resulting from experiments into .csv files for analysis

using HDF5, DataFrames, CSV, Glob, Dates, Random

"""

Create a DataFrame for a particular job. `nrows=2504` and `ncolumns=81271767` is correct for the 
1000 Genomes dataset.
"""
function create_df(fid, nrows=2504, ncolumns=81271767)
    rv = DataFrame()
    row = Dict{String, Any}()
    row["mse"] = missing # initialize to missing, it's computed later

    # store job parameters
    if "parameters" in keys(fid) && typeof(fid["parameters"]) <: HDF5.Group
        g = fid["parameters"]
        for key in keys(g)
            value = g[key][]
            row[key] = value
        end
    end

    # default values for nrows and ncolumns
    # (previous versions would not store these in the output file)
    if !haskey(row, "nrows")
        row["nrows"] = nrows
    end
    if !haskey(row, "ncolumns")
        row["ncolumns"] = ncolumns
    end    

    # add benchmark data
    niterations = Int(row["niterations"])
    nworkers = Int(row["nworkers"])    
    for i in 1:niterations
        row["iteration"] = i
        row["t_compute"] = fid["benchmark/t_compute"][i]
        row["t_update"] = fid["benchmark/t_update"][i]
        for j in 1:nworkers # worker response epochs
            row["repoch_worker_$j"] = fid["benchmark/responded"][j, i]
        end
        if "latency" in keys(fid["benchmark"]) # worker latency
            for j in 1:nworkers
                row["latency_worker_$j"] = fid["benchmark/latency"][j, i]
            end
        end            
        if "compute_latency" in keys(fid["benchmark"]) # worker compute latency
            for j in 1:nworkers
                row["compute_latency_worker_$j"] = fid["benchmark/compute_latency"][j, i]
            end
        end
        push!(rv, row, cols=:union)
    end
    rv
end

"""

Compute the explained variance (here referred to as mse). `Xnorm=104444.37027911078` is correct 
for the 1000 Genomes dataset.
"""
function compute_mse!(mses, iterates, Xs; mseiterations=20, Xnorm=104444.37027911078)
    if iszero(mseiterations)
        return mses
    end
    niterations = size(iterates, 3)
    is = unique(round.(Int, exp.(range(log(1), log(niterations), length=mseiterations))))
    for k in 1:length(is)
        i = is[k]
        if !ismissing(mses[i])
            continue
        end
        norms = zeros(length(Xs))
        t = @elapsed begin
            Threads.@threads for j in 1:length(Xs)
                norms[j] = norm(view(iterates, :, :, i)'*Xs[j])
            end
        end
        mses[i] = (sum(norms) / Xnorm)^2
        println("Iteration $i finished in $t s")
    end
    GC.gc()
    mses
end

"""
    df_from_output_file(filename::AbstractString, Xs::Vector{<:AbstractMatrix}; df_filename::AbstractString=replace(filename, ".h5"=>".csv"), mseiterations=0, reparse=false)

Parse the .h5 file `filename` (resulting from a run of the PCA kernel) into a DataFrame, which is 
written to disk as a .csv file with name `df_filename`. If `reparse = false` and `df_filename` 
already exists, then the existing `.csv` file is read from disk and returned, otherwise the `.h5`
file is parsed again.

To compute the explained variance of each iteration (here referred to as mse), the data matrix `X`
must be provided as a vector `Xs`, corresponding to a horizontal partitioning of the data matrix, 
i.e., `X = hcat(Xs, ...)`. Explained variance is computed for of up to `mseiterations` different 
iterations.
"""
function df_from_output_file(filename::AbstractString, Xs::Union{Nothing, Vector{<:AbstractMatrix}}; df_filename::AbstractString=replace(filename, ".h5"=>".csv"), mseiterations=0, reparse=false)
    # skip non-existing/non-hdf5 files
    if !HDF5.ishdf5(filename)
        println("skipping (not a HDF5 file): $filename")
        return DataFrame()
    end
    if !reparse && isfile(df_filename)
        return DataFrame(CSV.File(df_filename))
    end
    h5open(filename) do fid
        df = isfile(df_filename) ? DataFrame(CSV.File(df_filename)) : create_df(fid)
        df = df[.!ismissing.(df.iteration), :]
        if "iterates" in keys(fid) && mseiterations > 0
            sort!(df, :iteration)
            mses = Vector{Union{Float64,Missing}}(df.mse)
            select!(df, Not(:mse))
            df.mse = compute_mse!(mses, fid["iterates"][:, :, :], Xs; mseiterations)
        end
        CSV.write(df_filename, df)
        return df
    end
end


"""

Read the sparse matrix stored in dataset with `name` in `filename` and partitions it column-wise 
into `nblocks` partitions.
"""
function load_inputmatrix(filename::AbstractString, name::AbstractString="X"; nblocks=Threads.nthreads())
    h5open(filename) do fid
        m, n = h5size(fid, name)
        return [h5readcsc(fid, name, floor(Int, (i-1)/nblocks*n+1), floor(Int, i/nblocks*n)) for i in 1:nblocks]
    end
end

"""

Aggregate all DataFrames in `dir` into a single DataFrame.
"""
function aggregate_dataframes(;dir::AbstractString, prefix::AbstractString="output", dfname::AbstractString="df.csv")
    filenames = glob("$(prefix)*.csv", dir)
    println("Aggregating $(length(filenames)) files")
    dfs = [DataFrame(CSV.File(filename)) for filename in filenames]
    for (i, df) in enumerate(dfs)
        df[!, :jobid] .= i # store a unique ID for each file read
    end
    df = vcat(dfs..., cols=:union)
    df = clean_pca_df(df)
    CSV.write(joinpath(dir, dfname), df)
    df
end

"""

Return a vector composed of the number of flops performed by each worker and iteration. The density
of the 1000 Genomes data matrix is `0.05360388070027386`.
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

Cleanup
"""
function clean_pca_df(df::DataFrame)
    df = df[.!ismissing.(df.nworkers), :]
    df = df[.!ismissing.(df.iteration), :]
    df[!, :nostale] .= Missings.replace(df.nostale, false)
    df[!, :kickstart] .= Missings.replace(df.kickstart, false)
    df = df[df.kickstart .== false, :]
    select!(df, Not(:kickstart)) # drop the kickstart column
    df[!, :worker_flops] = worker_flops_from_df(df)
    df.npartitions = df.nworkers .* df.nsubpartitions
    rename!(df, :t_compute => :latency)
    rename!(df, :t_update => :update_latency)
    df[!, :nbytes] = df.nrows .* df.ncomponents .* 4 # Float32 entries => 4 bytes per entry
    sort!(df, [:jobid, :iteration])
    df.time = combine(groupby(df, :jobid), :latency => cumsum => :time).time # cumulative time since the start of the computation
    df.time .+= combine(groupby(df, :jobid), :update_latency => cumsum => :time).time
    df
end

"""

Read all output files from `dir` and write summary statistics (e.g., iteration time and convergence) to DataFrames.
"""
function parse_pca_files(;dir::AbstractString, prefix="output", dfname="df.csv", reparse=false, Xs=nothing, mseiterations=20)

    # process output files
    filenames = glob("$(prefix)*.h5", dir)
    shuffle!(filenames) # randomize the order to minimize overlap when using multiple concurrent processes
    for (i, filename) in enumerate(filenames)
        t = now()
        println("[$i / $(length(filenames)), $(Dates.format(now(), "HH:MM"))] parsing $filename")
        try
            df_from_output_file(filename, Xs; mseiterations, reparse)
        catch e
            printstyled(stderr,"ERROR: ", bold=true, color=:red)
            printstyled(stderr,sprint(showerror,e), color=:light_red)
            println(stderr)            
        end
        GC.gc()
    end
    inputmatrix = nothing
    GC.gc()
    aggregate_dataframes(;dir, prefix, dfname)
end


"""

Run `parse_pca_files` in a loop.
"""
function parse_loop(args...; kwargs...)
    while true
        GC.gc()
        parse_pca_files(args...; kwargs...)
        sleep(60)        
    end
end