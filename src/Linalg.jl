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
    explained_variance(X, V, den)

Explained variance with pre-computed denominator.
"""
explained_variance(X, V, Xnorm::Real) = min((norm(X*V) / Xnorm)^2, 1.0-eps(Float64))

"""
    explained_variance(X, V)

Return the fraction of variance explained by the principal components
in V, defined as tr(V'X'XV) / tr(X'X).

"""
explained_variance(X, V) = explained_variance(X, V, norm(X))

"""

Compute the right-singular vectors from the left-singular vectors `U` and the data matrix `X`.
"""
right_from_left_singular_vectors(X, U) = orthogonal!(U'*X)

"""

In-place version of the above.
"""
right_from_left_singular_vectors!(V, X, U) = orthogonal!(mul!(V, U', X))

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