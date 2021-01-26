using HDF5, DataFrames, CSV, Glob, Dates, Random
using CodedComputing

"""

Parse an output file and record everything in a DataFrame.
"""
function df_from_output_file(filename::AbstractString, inputmatrix)    
    t = now()
    println("[$(Dates.format(now(), "HH:MM"))] parsing $filename")

    # return a memoized result if one exists
    df_filename = filename * ".csv"
    if isfile(df_filename)
        return DataFrame(CSV.File(df_filename))
    end

    # skip non-existing/non-hdf5 files
    rv = DataFrame()
    if !HDF5.ishdf5(filename)
        println("skipping (not a HDF5 file): $filename")
        return rv
    end    

    h5open(filename) do fid        
        row = Dict{String,Any}()
        row["nrows"] = size(inputmatrix, 1)
        row["ncolumns"] = size(inputmatrix, 2)
        niterations = fid["parameters/niterations"][]
        nworkers = fid["parameters/nworkers"][]

        # store all parameters the job was run with
        if "parameters" in keys(fid) && typeof(fid["parameters"]) <: HDF5.Group
            g = fid["parameters"]
            for key in keys(g)
                value = g[key][]
                row[key] = value
            end
        end

        # compute mse using multiple threads (it's by far the most time-consuming part of the parsing)
        mses = Vector{Union{Missing,Float64}}(missing, niterations)
        if "iterates" in keys(fid)
            l = ReentrantLock() # HDF5 read isn't thread-safe
            cache = zeros(eltype(fid["iterates"]), size(fid["iterates"], 1), size(fid["iterates"], 2), min(Threads.nthreads(), niterations))
            Threads.@threads for i in 1:niterations
                j = Threads.threadid()                
                begin
                    lock(l)
                    try
                        cache[:, :, j] .= fid["iterates"][:, :, i]
                    finally
                        unlock(l)
                    end
                end
                mses[i] = explained_variance(inputmatrix, view(cache, :, :, j))
            end
        end

        # add benchmark data
        for i in 1:niterations
            row["iteration"] = i
            row["mse"] = mses[i]
            row["t_compute"] = fid["benchmark/t_compute"][i]
            row["t_update"] = fid["benchmark/t_update"][i]

            # add worker response epochs
            for j in 1:nworkers
                row["repoch_worker_$j"] = fid["benchmark/responded"][j, i]
            end

            push!(rv, row, cols=:union)
        end
    end    

    # memoize the resulting df
    CSV.write(df_filename, rv)

    rv
end

"""

Aggregate all DataFrames in `dir` into a single DataFrame.
"""
function aggregate_benchmark_dataframes(;dir::AbstractString, prefix::AbstractString="output", dfname::AbstractString="df.csv")
    dfs = [DataFrame(CSV.File(filename)) for filename in glob("$(prefix)*.csv", dir)]
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
function parse_benchmark_files(;dir::AbstractString, inputfile::AbstractString, inputname="X", prefix="output", dfname="df.csv")

    # read input matrix (used to measure convergence)
    iscsc = false
    h5open(inputfile) do fid
        iscsc, _ = isvalidh5csc(fid, inputname)
    end
    if iscsc
        X = h5readcsc(inputfile, inputname)
    else
        X = h5read(inputfile, inputname)
    end

    # process output files
    filenames = glob("$(prefix)*.h5", dir)
    shuffle!(filenames) # randomize the order to minimize overlap when using multiple concurrent processes
    for filename in filenames
        try
            df_from_output_file(filename, X) # the result is memoized on disk
        catch e
            printstyled(stderr,"ERROR: ", bold=true, color=:red)
            printstyled(stderr,sprint(showerror,e), color=:light_red)
            println(stderr)            
        end
        GC.gc()
    end
    aggregate_benchmark_dataframes(;dir, prefix, dfname)
end

# if run as a script
if abspath(PROGRAM_FILE) == @__FILE__
    dir = ARGS[1]
    inputfile = ARGS[2]
    parse_benchmark_files(;dir, inputfile)
end