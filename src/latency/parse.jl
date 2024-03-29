# Parse the output of the latency script into a DataFrame
export df_from_latency_file

"""

Parse an output file and record everything in a DataFrame.
"""
function df_from_latency_file(filename::AbstractString)

    # skip non-existing/non-hdf5 files
    rv = DataFrame()
    if !HDF5.ishdf5(filename)
        println("skipping (not a HDF5 file): $filename")
        return rv
    end    

    h5open(filename) do fid        
        row = Dict{String,Any}()
        niterations = fid["parameters/niterations"][]
        nworkers = fid["parameters/nworkers"][]

        # record parameters
        if "parameters" in keys(fid) && typeof(fid["parameters"]) <: HDF5.Group
            g = fid["parameters"]
            for key in keys(g)
                value = g[key][]
                row[key] = value
            end
        end

        # record timestamps and latency
        for i in 1:niterations
            row["iteration"] = i
            row["latency"] = fid["latency"][i]
            row["timestamp"] = fid["timestamps"][i]
            for j in 1:nworkers
                row["repoch_worker_$j"] = fid["worker_repochs"][j, i]
                row["latency_worker_$j"] = fid["worker_latency"][j, i]
                row["compute_latency_worker_$j"] = fid["worker_compute_latency"][j, i]
            end
            push!(rv, row, cols=:union)
        end
    end    
    rv
end

"""

Read a latency experiment csv file into a DataFrame
"""
function clean_latency_df(df::DataFrame)
    df.worker_flops = 2 .* df.nrows .* df.ncols .* df.ncomponents .* df.density
    sort!(df, [:jobid, :iteration])
    df.time = by(df, :jobid, :latency => cumsum => :time).time # cumulative time since the start of the computation
    df
end

"""

Read all output files from `dir` and write summary statistics (e.g., iteration time and convergence) to DataFrames.
"""
function parse_latency_files(;dir::AbstractString, prefix="output", dfname="df.csv")

    # process output files
    filenames = glob("$(prefix)*.h5", dir)
    for filename in filenames
        println("Parsing $filename")
        try
            df = df_from_latency_file(filename)
            CSV.write(replace(filename, ".h5"=>".csv"), df)
            rm(filename)
        catch e
            printstyled(stderr,"ERROR: ", bold=true, color=:red)
            printstyled(stderr,sprint(showerror,e), color=:light_red)
            println(stderr)            
        end
        GC.gc()
    end
    df = aggregate_dataframes(;dir, prefix, dfname)
    clean_latency_df(df)
end

function latency_parse_loop(args...; kwargs...)
    while true
        GC.gc()
        parse_latency_files(args...; kwargs...)
        sleep(60)        
    end
end