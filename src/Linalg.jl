# Linear algebra functions not in the standard library.

export orthogonal, orthogonal!, pca, explained_variance

"""Return the angle between a and b"""
function Base.angle(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) || throw(DimensionMismatch("a has dimension $(length(a)), but b has dimension $(length(b))"))
    acos(min(max(-1.0, dot(a, b) / norm(a) / norm(b)), 1.0))
end

"""return the angle between w and v, accounting for their sign"""
minangle(w, v) = min(angle(w, v), angle(w, -v))

"""Compute PCA via the built-in SVD."""
function pca(X, k::Integer=size(X, 2))
    F = svd(X)
    return F.V[:, 1:k]
end

"""Orthogonalize and normalize the columns of A in-place."""
function orthogonal!(A::AbstractMatrix)
    m, n = size(A)
    for i in 1:n
        for j in 1:i-1
            l = dot(view(A, :, j), view(A, :, i))
            view(A, :, i) .-= l.*view(A, :, j)
        end
        view(A, :, i) ./= norm(view(A, :, i))
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