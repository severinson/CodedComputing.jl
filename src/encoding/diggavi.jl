# Source data encoding
# An implementation of [Diggavi, 2017]
#
# Notation:
# m is the total number of workers
# k is the number of workers to wait for
# η = k/m (fraction of workers to wait for)
# n is the number of samples (rows of X)
# p is the sample dimension (columns of X)
# β is the redundancy factor
# μ is the smallest eigenvalue of X'*X
# M is the largest eigenvalue of X'*X

using Hadamard, Random

"""
    hadamard_generator_matrix(nc::Integer, k::Integer)

Return a generator matrix of size `(nc, k)` created by sampling the columns of 
a Hadamard matrix. Always returns the same matrix for a given `(nc, k)` pair.
This function is only meant for testing; use the encode_hadamard function
for encoding.
"""
function hadamard_generator_matrix(nc::Integer, k::Integer)
    
    # create a Hadamard matrix of size equal to the next power of 2 from nc
    np = 2^ceil(Int, log2(nc))
    F = float.(hadamard(np))

    # create a (nc, k) generator matrix by selecting the first nc rows 
    # and k columns, selected at random
    p = collect(1:np)
    shuffle!(MersenneTwister(k), p)
    S = view(F, 1:nc, view(p, 1:k))

    # normalize to ensure eigenvalues are centered around 1
    S ./= sqrt(np)
end

"""
    encode_hadamard!(c::AbstractArray{T,N}, x::AbstractArray{T,N}, nc::Integer) where {T,N}

In-place version of `encode_hadamard`. The first dimension of `c` has to be a power of 2.
This function returns a view into the first `nc` rows of `c`.
"""
function encode_hadamard!(c::AbstractArray{T,N}, x::AbstractArray{T,N}, nc::Integer) where {T,N}
    nc >= size(x, 1) || throw(DomainError(nc, "nc is $nc, but x has dimensions $(size(x))"))
    np = 2^ceil(Int, log2(nc)) # next power of 2 from nc
    np == size(c, 1) || throw(DomainError(np, "c has dimensions $(size(c))"))
    p = collect(1:np)
    shuffle!(MersenneTwister(size(x, 1)), p)
    selectdim(c, 1, view(p, 1:size(x, 1))) .= x
    fwht_natural!(c, 1)
    c .*= sqrt(np) # ensures the spectrum of SA'*SA is <= 1
    selectdim(c, 1, 1:nc)
end

"""
    encode_hadamard(x::AbstractArray, nc::Integer)

Encode the array `c` along its first dimension using a fast Hadamard transform.
Returns a new array, for the first dimension is `nc`.
"""
function encode_hadamard(x::AbstractArray, nc::Integer)
    np = 2^ceil(Int, log2(nc)) # next power of 2 from n
    encode_hadamard!(zeros(eltype(x), np, size(x)[2:end]...), x, nc)
end