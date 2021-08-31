using CodedComputing
using Random, MPI, HDF5, H5Sparse, LinearAlgebra, SparseArrays
using Dates
using Test

"""

Return an array composed of the PCA computed iterates.
"""
function load_pca_iterates(outputfile::AbstractString, outputdataset::AbstractString)
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test outputdataset in keys(fid)
        if "iterates" in keys(fid) && length(size(fid["iterates"])) == 3 && size(fid["iterates"], 3) > 0
            @test fid[outputdataset][:, :] ≈ fid["iterates"][:, :, end]
            return [fid["iterates"][:, :, i] for i in 1:size(fid["iterates"], 3)]
        else # return only the final iterate if intermediate iterates aren't stored
            return [fid[outputdataset][:, :]]
        end
    end
end

function load_logreg_iterates(outputfile::AbstractString, outputdataset::AbstractString)
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test outputdataset in keys(fid)
        if "iterates" in keys(fid) && length(size(fid["iterates"])) == 2 && size(fid["iterates"], 2) > 0
            @test fid[outputdataset][:] ≈ fid["iterates"][:, end]
            return [fid["iterates"][:, i] for i in 1:size(fid["iterates"], 2)]
        else # return only the final iterate if intermediate iterates aren't stored
            return [fid[outputdataset][:]]
        end
    end
end

function test_pca_iterates(;X::AbstractMatrix, niterations::Integer, ncomponents::Integer, 
                            ev::Real, outputfile::AbstractString, outputdataset::AbstractString, atol=1e-2)
    dimension, nsamples = size(X)
    Vs = load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    @test length(Vs) == niterations
    @test all((V)->size(V)==(dimension, ncomponents), Vs)
    @test Vs[end]'*Vs[end] ≈ I
    @test fs[end] < ev || isapprox(fs[end], ev, atol=atol)
    return Vs, fs
end

function logreg_loss(v, X, b, λ)
    rv = 0.0
    for i in 1:length(b)
        rv += log(1 + exp(-b[i]*(v[1]+dot(X[:, i], view(v, 2:length(v))))))
    end
    rv / length(b) + λ/2 * norm(v)^2
end

# @testset "latency.jl" begin
#     kernel = "../src/latency/kernel.jl"
#     nwait = 1
#     nworkers = 3
#     niterations = 10
#     timeout = 0.1
#     nbytes = 100
#     nrows = 100
#     ncols = 100
#     ncomponents = 3
#     density = 0.1
#     outputfile = tempname()
#     mpiexec(cmd -> run(```
#         $cmd -n $(nworkers+1) julia --project $kernel $outputfile
#             --niterations $niterations
#             --nbytes $nbytes
#             --nrows $nrows
#             --ncols $ncols
#             --ncomponents $ncomponents
#             --density $density
#             --nwait $nwait
#             --timeout $timeout            
#     ```))
#     df = df_from_latency_file(outputfile)
#     @test all(diff(df.timestamp) .>= timeout)
# end

@time @testset "Linalg.jl" begin include("linalg_test.jl") end
@time @testset "eventdriven.jl" begin include("eventdriven_test.jl") end
@time @testset "profiling.jl" begin include("profiling_test.jl") end
@time @testset "loadbalancing.jl" begin include("loadbalancing_test.jl") end
@time @testset "mul.jl" begin include("mul_test.jl") end
@time @testset "logreg.jl" begin include("logreg_test.jl") end
@time @testset "pca.jl" begin include("pca_test.jl") end