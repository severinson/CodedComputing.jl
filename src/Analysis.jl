using CSV, DataFrames, PyPlot, Statistics, Polynomials, LinearAlgebra, Distributions, RollingFunctions
using StatsBase

using PyCall
tikzplotlib = pyimport("tikzplotlib")

"""

Fit a linear model (i.e., a line) to the data X, y.
"""
function linear_model(X::AbstractMatrix, y::AbstractVector)
    size(X, 1) == length(y) || throw(DimensionMismatch("X has dimensions $(size(X)), but y has dimension $(length(y))"))
    A = ones(size(X, 1), size(X, 2)+1)    
    A[:, 2:end] .= X
    A \ y
end

"""

Fit a polynomial of degree `d` to the data `xs` and `ys`
"""
function fit_polynomial(xs::AbstractVector, ys::AbstractVector, d=1)
    A = ones(length(xs), d+1)
    for i in 1:d
        A[:, i+1] .= xs.^i
    end
    try
        coeffs = A\ys
        return (x) -> dot(coeffs, [x^i for i in 0:d]), coeffs        
    catch SingularException
        return (x) -> 0, fill(NaN, d+1)
    end
end

linear_model(x::AbstractVector, y::AbstractVector) = linear_model(reshape(x, length(x), 1), y)

"""

Write xs and ys as a table with columns separated by a space
"""
function write_table(xs::AbstractVector, ys::AbstractVector, filename::AbstractString)
    length(xs) == length(ys) || throw(DimensionMismatch("xs has dimension $(length(xs)), but ys has dimension $(length(ys))"))
    open(filename, "w") do io
        for i in 1:length(xs)
            write(io, "$(xs[i]) $(ys[i])\n")
        end
    end
    return
end

includet("analysis/convergence.jl")
includet("analysis/correlation.jl")
includet("analysis/latency.jl")
includet("analysis/preprocessing.jl")