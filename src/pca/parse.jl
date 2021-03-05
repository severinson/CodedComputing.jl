# Code for parsing the .h5 files resulting from experiments into .csv files for analysis

using HDF5, DataFrames, CSV, Glob, Dates, Random

"""

Parse an output file and record everything in a DataFrame.
"""
function df_from_output_file(filename::AbstractString; inputfile::AbstractString, inputname::AbstractString, Xnorm::Real, mseiterations::Integer=2)

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
        nrows, ncolumns = h5size(inputfile, inputname)
        row["nrows"] = nrows
        row["ncolumns"] = ncolumns
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

        # compute explained variance        
        mses = Vector{Union{Missing,Float64}}(missing, niterations)
        if "iterates" in keys(fid)
            U = zeros(eltype(fid["iterates"]), size(fid["iterates"], 1), size(fid["iterates"], 2))
            for j in 1:mseiterations
                i = round(Int, j/mseiterations*niterations)
                println("Iteration $i / $niterations ($(j / mseiterations))")
                @time U .= fid["iterates"][:, :, i]
                @time UtX = h5mulcsc(U', inputfile, inputname)
                @time mses[i] = (norm(UtX) / Xnorm)^2
                println()
                GC.gc()
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

            # add worker latency
            if "latency" in names(fid["benchmark"])
                for j in 1:nworkers
                    row["latency_worker_$j"] = fid["benchmark/latency"][j, i]
                end
            end            

            push!(rv, row, cols=:union)
            GC.gc()
        end
    end    

    # memoize the resulting df
    CSV.write(df_filename, rv)

    rv
end

"""

Aggregate all DataFrames in `dir` into a single DataFrame.
"""
function aggregate_dataframes(;dir::AbstractString, prefix::AbstractString="output", dfname::AbstractString="df.csv")
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
function parse_pca_files(;dir::AbstractString, inputfile::AbstractString, inputname="X", prefix="output", dfname="df.csv", Xnorm=104444.37027911078)

    # process output files
    filenames = glob("$(prefix)*.h5", dir)
    shuffle!(filenames) # randomize the order to minimize overlap when using multiple concurrent processes
    for filename in filenames
        t = now()
        println("[$(Dates.format(now(), "HH:MM"))] parsing $filename")
        try
            df = df_from_output_file(filename; inputfile, inputname, Xnorm)
            CSV.write(filename*".csv", df)
            # rm(filename)
        catch e
            printstyled(stderr,"ERROR: ", bold=true, color=:red)
            printstyled(stderr,sprint(showerror,e), color=:light_red)
            println(stderr)            
        end
        GC.gc()
    end
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