using HDF5, DataFrames, CSV, Glob, Dates
using CodedComputing

function parse_output_file(filename::AbstractString, inputmatrix)    
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

        # compute mse
        mses = Vector{Union{Missing,Float64}}(missing, niterations)
        if "iterates" in keys(fid)
            iterates = [fid["iterates"][:, :, i] for i in 1:niterations]        
            Threads.@threads for i in 1:niterations
                mses[i] = explained_variance(inputmatrix, iterates[i])
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

Read all output files from a given directory and write summary statistics (e.g., iteration time 
and convergence) to a DataFrame.
"""
function aggregate_benchmark_data(;dir="/shared/201124/3/", inputfile="/shared/201124/ratings.h5", inputname="X", prefix="output", dfname="df.csv")

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
    dfs = Vector{DataFrame}(undef, length(filenames))
    for (i, filename) in collect(enumerate(filenames))
        try
            dfs[i] = parse_output_file(filename, X)
            dfs[i][:jobid] = i # store a unique ID for each file read
        catch e
            printstyled(stderr,"ERROR: ", bold=true, color=:red)
            printstyled(stderr,sprint(showerror,e), color=:light_red)
            println(stderr)            
            dfs[i] = DataFrame()
        end
    end

    # read all dfs from disk (so that we get any files without an associated .h5 file),
    # write the aggregated df to disk, and return
    dfs = [DataFrame(CSV.File(filename)) for filename in glob("$(prefix)*.csv", dir)]
    for (i, df) in enumerate(dfs)
        df[:jobid] = i # store a unique ID for each file read
    end
    df = vcat(dfs..., cols=:union)
    CSV.write(joinpath(dir, dfname), df)

    df
end