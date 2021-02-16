# Linear algebra functions not in the standard library.

export orthogonal, orthogonal!, pca, explained_variance, projection_distance

"""Return the angle between a and b"""
function Base.angle(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) || throw(DimensionMismatch("a has dimension $(length(a)), but b has dimension $(length(b))"))
    acos(min(max(-1.0, dot(a, b) / norm(a) / norm(b)), 1.0))
end

"""return the angle between w and v, accounting for their sign"""
minangle(w, v) = min(angle(w, v), angle(w, -v))

"""Compute PCA via the built-in SVD."""
function pca(X, k::Integer=min(size(X)...))
    F = svd(X)
    return F.V[:, 1:k]
end

"""Orthogonalize and normalize the columns of A in-place."""
function orthogonal!(A::AbstractMatrix)
    m, n = size(A)
    for i in 1:n
        for j in 1:i-1
            l = dot(view(A, :, j), view(A, :, i))
            for k in 1:size(A, 1)
                A[k, i] -= l*A[k, j]
            end
        end
        g = norm(view(A, :, i))
        if g > sqrt(eps(eltype(A)) * size(A, 1))
            view(A, :, i) ./= g
        end
        replace!(view(A, :, i), NaN=>zero(eltype(A)))
    end
    return A
end

"""Orthogonalize and normalize the columns of A."""
orthogonal(A::AbstractMatrix) = orthogonal!(copy(A))

"""
    explained_variance(X, V)

Return the fraction of variance explained by the principal components
in V, defined as tr(V'X'XV) / tr(X'X).

"""
function explained_variance(X, V)
    n, d = size(X)
    _, k = size(V)
    XV = X*V
    num = 0.0
    @inbounds for i in 1:k
        for j in 1:n
            num += Float64(XV[j, i])^2
        end
    end
    den = 0.0
    @inbounds for i in 1:d
        for j in 1:n
            den += Float64(X[j, i])^2
        end
    end
    min(num / den, 1.0-eps(Float64))
end

projection_distance(X, V) = sqrt(norm(X .- (X*V)*V')) / reduce(*, size(X))

function projection_distance(X::SparseMatrixCSC, V)
    rv = 0.0
    L = X*V
    R = V'    
    Is, Js, Vs = findnz(X)
    for (i, j, v) in zip(Is, Js, Vs)
        rv += (Float64(v) - dot(view(L, i, :), view(R, :, j)))^2
    end
    rv = sqrt(rv)
    rv /= length(V)
end

"""
    matrix_from_tensor(T)

Convert a tensor (i.e., a 3D matrix) to a matrix by flattening the first two dimensions. For 
example, if the tensor corresponds to a vector of images, each image corresponds to a row of 
the resulting matrix.
"""
function matrix_from_tensor(T)
    d1, d2, n = size(T)
    d = d1*d2
    Matrix(reshape(T, d, n)')
end