# Parse the output of the latency script into a DataFrame
export df_from_latency_file

"""

Parse an output file and record everything in a DataFrame.
"""
function df_from_latency_file(filename::AbstractString)


    # return a memoized result if one exists
    # df_filename = filename * ".csv"
    # if isfile(df_filename)
    #     return DataFrame(CSV.File(df_filename))
    # end

    # skip non-existing/non-hdf5 files
    rv = DataFrame()
    if !HDF5.ishdf5(filename)
        println("skipping (not a HDF5 file): $filename")
        return rv
    end    

    h5open(filename) do fid        
        row = Dict{String,Any}()

        # row["nrows"] = size(inputmatrix, 1)
        # row["ncolumns"] = size(inputmatrix, 2)
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
            end
            push!(rv, row, cols=:union)
        end
    end    

    # memoize the resulting df
    # CSV.write(df_filename, rv)

    rv
end