# Specialized routines for multiplying a subset of the columns of a sparse matrix by a vector or matrix

export colsmul!

function colsmul!(C::AbstractMatrix, A::SparseArrays.AbstractSparseMatrixCSC, B::AbstractMatrix, cols, α=1, β=0)
    length(size(C)) == length(size(B)) || throw(DimensionMismatch("C has dimensions $(size(C)), but B has dimensions $(size(B))"))
    size(C, 1) == size(A, 1) || throw(DimensionMismatch("C has dimensions $(size(C)), but A has dimensions $(size(A))"))
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("A has dimensions $(size(A)), but B has dimensions $(size(B))"))    
    0 < minimum(cols) <= maximum(cols) <= size(A, 2) || throw(ArgumentError("A has dimensions $(size(A)), but cols is $cols"))
    C .*= β
    rows = rowvals(A)
    vals = nonzeros(A)
    for col in cols
        for i in nzrange(A, col)
            @inbounds row = rows[i]
            @inbounds val = α * vals[i]
            for k in 1:size(C, 2)
                @inbounds C[row, k] += val*B[col, k]
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