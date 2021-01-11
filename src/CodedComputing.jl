module CodedComputing

using LinearAlgebra, SparseArrays, HDF5

include("Linalg.jl")
include("Datasets.jl")
include("HDF5Sparse.jl")

include("encoding/diggavi.jl")
include("analysis/pca.jl")
include("analysis/pcaplots.jl")
include("pca/benchmark.jl")

end
