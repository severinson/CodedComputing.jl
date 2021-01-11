using MLDatasets

"""
    load_dataset(n, d, f; filename::AbstractString)

Load a dataset by calling the function f, which is assumed to return an array, reshape that array
into a matrix of size `n` by `d` and return it. If filename is a string, the array is cached on 
disk (in a file with the given filename) to speed up subsequent calls.
"""
function load_dataset(n, d, f; filename::Union{Nothing,<:AbstractString})
    try # try to load from cache
        return Matrix(reshape(reinterpret(UInt8, read(filename)), n, d))
    catch SystemError # load from MLDatasets (about 10x slower)
        T = f()
        X = UInt8.(255 .* matrix_from_tensor(T))
        if !isnothing(filename)
            write(filename, X) # cache on disk to speed up subsequent calls
        end
        return X
    end
end

load_mnist_training() = load_dataset(60000, 784, MLDatasets.MNIST.traintensor, filename="MNIST_training.bin")
load_mnist_testing() = load_dataset(10000, 784, MLDatasets.MNIST.testtensor, filename="MNIST_testing.bin")