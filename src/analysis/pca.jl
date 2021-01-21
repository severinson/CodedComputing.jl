using HDF5, DataFrames, CSV, Glob
using CodedComputing

function parse_output_file(filename::AbstractString, inputmatrix)    
    println("parsing $filename")    

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

    # concatenate, write to disk, and return
    df = vcat(dfs..., cols=:union)
    CSV.write(joinpath(dir, dfname), df)
    df
end

"""

Return a matrix of size `n` by `m`, with `dimension` singular values of value `σ1` and 
`m-dimension` singular values of value `σ2`.
"""
function pca_test_matrix1(n::Integer, m::Integer, dimension::Integer; σ1=1.0, σ2=0.1)
    n > 0 || throw(DomainError(n, "n must positive"))
    0 < m <= n || throw(DomainError(m, "m must be in [1, n]"))
    dimension > 0 || throw(DomainError(dimension, "dimension must positive"))
    dimension <= m || throw(DimensionMismatch("dimension is $dimension, but m is $m"))
    U = orthogonal!(randn(n, m))
    V = orthogonal!(randn(m, m))
    S = append!(repeat([σ1], dimension), repeat([σ2], m-dimension))
    U*Diagonal(S)*V'
end

"""

Return a matrix of size `n` by `m`, where the rows correspond to points embedded in a hyperplane
of dimension `dimension`.
"""
function pca_test_matrix2(n::Integer, m::Integer, dimension::Integer)
    n > 0 || throw(DomainError(n, "n must positive"))
    0 < m <= n || throw(DomainError(m, "m must be in [1, n]"))
    dimension > 0 || throw(DomainError(dimension, "dimension must positive"))
    dimension <= m || throw(DimensionMismatch("dimension is $dimension, but m is $m"))    
    G = orthogonal!(randn(m, dimension))
    P = G*G' # projection matrix
    randn(n, m)*P .+ randn(n, m) .* σ
end

"""

Convert a DataFrame containing MovieLens ratings into a ratings matrix, where rows correspond to
users, columns to movies, and the `[i, j]`-th entry is the rating given by user `i` to movie `j`.
"""
function movielens_rating_matrix(df::DataFrame)
    movieIds = unique(df.movieId)
    userIds = unique(df.userId)
    nmovies = length(movieIds)
    nusers = length(userIds)
    movie_perm = collect(1:maximum(df.movieId))
    user_perm = collect(1:maximum(df.userId))
    movie_perm[movieIds] .= 1:nmovies
    user_perm[userIds] .= 1:nusers
    Is = user_perm[df.userId]
    Js = movie_perm[df.movieId]
    Vs = Int8.(df.rating .* 2)
    sparse(Is, Js, Vs, nusers, nmovies)    
end

function movielens_rating_matrix(filename="/shared/MovieLens/ml-25m/ratings.csv")
    df = DataFrame(CSV.File(filename, normalizenames=true))
    movielens_rating_matrix(df)
end