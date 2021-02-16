module CodedComputing

using LinearAlgebra, SparseArrays, HDF5

include("Linalg.jl")
include("Datasets.jl")
include("HDF5Sparse.jl")
include("OrderStats.jl")

include("latency/parse.jl")
include("latency/benchmark.jl")

include("encoding/diggavi.jl")
include("analysis/parse.jl")
include("pca/benchmark.jl")

end
