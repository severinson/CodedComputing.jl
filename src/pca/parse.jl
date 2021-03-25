# Code for parsing the .h5 files resulting from experiments into .csv files for analysis

using HDF5, DataFrames, CSV, Glob, Dates, Random

function create_df(fid, nrows=2504, ncolumns=81271767)
    rv = DataFrame()
    row = Dict{String, Any}()
    row["nrows"] = nrows
    row["ncolumns"] = ncolumns
    niterations = fid["parameters/niterations"][]
    nworkers = fid["parameters/nworkers"][]
    row["mse"] = missing # initialize to missing, it's computed later

    # store job parameters
    if "parameters" in keys(fid) && typeof(fid["parameters"]) <: HDF5.Group
        g = fid["parameters"]
        for key in keys(g)
            value = g[key][]
            row[key] = value
        end
    end

    # add benchmark data
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
        push!(rv, row, cols=:union)
    end
    rv
end

function compute_mse!(mses, iterates, Xs; mseiterations=0, Xnorm=104444.37027911078)
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

Parse an output file and record everything in a DataFrame.
"""
function df_from_output_file(filename::AbstractString, Xs; df_filename::AbstractString=filename*".csv", mseiterations=0, reparse=false)
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

# 2504×81271767, about 100GB total
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
        df[:jobid] = i # store a unique ID for each file read
    end
    df = vcat(dfs..., cols=:union)
    CSV.write(joinpath(dir, dfname), df)
    df
end

"""

Read all output files from `dir` and write summary statistics (e.g., iteration time and convergence) to DataFrames.
"""
function parse_pca_files(;dir::AbstractString, prefix="output", dfname="df.csv", reparse=false, Xs, mseiterations=0)

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

function parse_loop(args...; kwargs...)
    while true
        GC.gc()
        parse_benchmark_files(args...; kwargs...)        
        sleep(60)        
    end
end

# if run as a script
if abspath(PROGRAM_FILE) == @__FILE__
    dir = ARGS[1]
    inputfile = ARGS[2]
    parse_benchmark_files(;dir, inputfile)
end