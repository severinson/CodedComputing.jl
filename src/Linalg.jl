# Linear algebra functions not in the standard library.

export orthogonal, orthogonal!, pca

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