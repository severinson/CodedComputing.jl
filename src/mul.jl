# Specialized routines for multiplying a subset of the columns of a sparse matrix by a vector or matrix

export colsmul!, tcolsmul!

"""
    colsmul!(C::AbstractMatrix, A::SparseArrays.AbstractSparseMatrixCSC, B::AbstractMatrix, cols, α=1, β=0)

Efficient implementation of the multiplication `α .* A[:, cols] * B[cols, :] .+ β .* C`, where `A` 
is sparse. The result is stored in-place in `C`.
"""
function colsmul!(C::AbstractMatrix, A::SparseArrays.AbstractSparseMatrixCSC, B::AbstractMatrix, cols, α=1, β=0)
    length(size(C)) == length(size(B)) || throw(DimensionMismatch("C has dimensions $(size(C)), but B has dimensions $(size(B))"))
    size(C, 1) == size(A, 1) || throw(DimensionMismatch("C has dimensions $(size(C)), but A has dimensions $(size(A))"))
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("A has dimensions $(size(A)), but B has dimensions $(size(B))"))    
    0 < minimum(cols) <= maximum(cols) <= size(A, 2) || throw(ArgumentError("A has dimensions $(size(A)), but cols is $cols"))
    C .*= β
    rows = rowvals(A)
    vals = nonzeros(A)
    for col in cols
        for k in 1:size(C, 2)
            @inbounds v = α * B[col, k]
            for i in nzrange(A, col)
                @inbounds row = rows[i]
                @inbounds val = vals[i]
                v1 = val * v
                @inbounds C[row, k] += v1
            end
        end
    end
    C
end

function colsmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, cols, args...)
    mul!(C, view(A, :, cols), view(B, cols, :), args...)
end

function colsmul!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector, args...)
    colsmul!(reshape(c, length(c), 1), A, reshape(b, length(b), 1), args...)
    c
end

"""

Efficient implementation of the multiplication `α .* Transpose(A[:, cols]) * B .+ β .* C`, where 
`A` is sparse. The result is stored in-place in `C[cols, :]`.
"""
function tcolsmul!(C::AbstractMatrix, A::SparseArrays.AbstractSparseMatrixCSC, B::AbstractMatrix, cols, α=1, β=0)
    length(size(C)) == length(size(B)) || throw(DimensionMismatch("C has dimensions $(size(C)), but B has dimensions $(size(B))"))
    size(C, 1) == size(A, 2) || throw(DimensionMismatch("C has dimensions $(size(C)), but A has dimensions $(size(A))"))
    size(A, 1) == size(B, 1) || throw(DimensionMismatch("A has dimensions $(size(A)), but B has dimensions $(size(B))"))    
    0 < minimum(cols) <= maximum(cols) <= size(A, 2) || throw(ArgumentError("A has dimensions $(size(A)), but cols is $cols"))
    rows = rowvals(A)
    vals = nonzeros(A)
    C .*= β
    for col in cols
        for k in 1:size(C, 2)
            v1 = zero(eltype(C))
            for i in nzrange(A, col)
                @inbounds row = rows[i]
                @inbounds val = vals[i]
                @inbounds v2 = B[row, k]
                v1 += val*v2
            end
            v1 *= α
            @inbounds C[col, k] += v1
        end
    end
    C
end

function tcolsmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, cols, args...)
    @views mul!(C[cols, :], transpose(A[:, cols]), B, args...)
end

function tcolsmul!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector, args...)
    colsmul!(reshape(c, length(c), 1), A, reshape(b, length(b), 1), args...)
    c
end