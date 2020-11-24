using HDF5, DataFrames, CSV, Glob
using CodedComputing

"""

Read all output files from a given directory and write summary statistics (e.g., iteration time 
and convergence) to a DataFrame.
"""
function aggregate_benchmark_data(;dir="/shared/201123/2/", inputfile="/shared/201123/input.h5", prefix="output", dfname="df.csv")
    t_compute_all = zeros(Float64, 0)
    t_update_all = zeros(Float64, 0)
    nworkers_all = zeros(Int, 0)
    nwait_all = zeros(Int, 0)
    iteration_all = zeros(Int, 0)
    jobid_all = zeros(Int, 0)
    jobid = 1
    responded_all = zeros(Bool, 0, 0)

    explained_variance_all = zeros(Union{Float64,Missing}, 0)
    X = h5read(inputfile, "X")
    for filename in glob("$(prefix)*.h5", dir)
        println(filename)
        if !HDF5.ishdf5(filename)
            continue
        end
        h5open(filename) do fid
            nwait = fid["parameters/nwait"][]
            nworkers = fid["parameters/nworkers"][]
            n = length(fid["benchmark"]["t_compute"])
            @assert n == length(fid["benchmark/t_update"])
            append!(t_compute_all, fid["benchmark/t_compute"][:])
            append!(t_update_all, fid["benchmark/t_update"][:])
            append!(nworkers_all, repeat([nworkers], n))
            append!(nwait_all, repeat([nwait], n))
            append!(iteration_all, 1:n)
            append!(jobid_all, repeat([jobid], n))
            jobid += 1
            if "iterates" in keys(fid)
                append!(
                    explained_variance_all, 
                    [explained_variance(X, fid["iterates"][:, :, i]) for i in 1:n],
                )
            else
                append!(explained_variance_all, repeat([missing], n))
            end
            responded_all = vcat(responded_all, zeros(Bool, n, size(responded_all, 2)))
            if "responded" in keys(fid["benchmark"])
                if nworkers > size(responded_all, 2)
                    responded_all = hcat(
                        responded_all, 
                        zeros(Bool, size(responded_all, 1), nworkers-size(responded_all, 2))
                        )
                end                
                responded_all[end-n+1:end, 1:nworkers] .= fid["benchmark/responded"][:, :]'
            end
        end
    end
    df = DataFrame(
        iteration=iteration_all, 
        nworkers=nworkers_all, 
        nwait=nwait_all, 
        t_compute=t_compute_all, 
        t_update=t_update_all,
        explained_variance=explained_variance_all,
        jobid=jobid_all,
        )
    for i in 1:size(responded_all, 2)
        df["worker_$(i)_responded"] = responded_all[:, i]
    end
    CSV.write(dir*dfname, df)
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