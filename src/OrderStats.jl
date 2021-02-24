using Random, Statistics, Distributions
export ExponentialOrder, OrderStatistic, NonIDOrderStatistic, TukeyLambda

"""
    ExponentialOrder(scale::Real, total::Int, order::Int)

Random variable representing the order-th largest value out of total
realizations of an exponential random variable with given scale.

"""
function ExponentialOrder(scale::Real, total::Int, order::Int)
    scale > 0 || throw(DomainError(scale, "scale must be positive"))
    total > 0 || throw(DomainError((total, order), "total must be positive"))
    order > 0 || throw(DomainError((total, order), "order must be positive"))
    order <= total || throw(DomainError((total, order), "order must be <= total"))
    var = sum(1/(i^2)  for i=(total-order+1):total) * scale^2
    mean = sum(1/i for i=(total-order+1):total) * scale
    alpha = mean^2 / var # shape parameter
    theta = var / mean # scale
    return Gamma(alpha, theta)
end

struct OrderStatistic{S<:Union{Discrete,Continuous},Spl<:Sampleable{Univariate,S},T} <: Sampleable{Univariate,S}
    spl::Spl
    k::Int
    buffer::Vector{T}
end

Base.show(io::IO, s::OrderStatistic) = print(io, "OrderStatistic($(s.spl), k=$(s.k), n=$(length(s.buffer)))")

OrderStatistic(s::Sampleable, k::Integer, n::Integer) = OrderStatistic(s, k, Vector{eltype(s)}(undef, n))

function Random.rand(rng::AbstractRNG, s::OrderStatistic)
    Distributions.rand!(rng, s.spl, s.buffer)
    partialsort!(s.buffer, s.k)
    s.buffer[s.k]
end

struct NonIDOrderStatistic{S<:Union{Discrete,Continuous},Spl<:Sampleable{Univariate,S},T} <: Sampleable{Univariate,S}
    spls::Vector{Spl}
    k::Int
    buffer::Vector{T}
end

Base.show(io::IO, s::NonIDOrderStatistic) = print(io, "NonIDOrderStatistic($(eltype(s.spls)), k=$(s.k), n=$(length(s.buffer)))")

NonIDOrderStatistic(spls::AbstractVector{<:Sampleable}, k::Integer) = NonIDOrderStatistic(spls, k, Vector{promote_type((eltype(spl) for spl in spls)...)}(undef, length(spls)))

function Random.rand(rng::AbstractRNG, s::NonIDOrderStatistic)
    for (i, spl) in enumerate(s.spls)
        s.buffer[i] = Distributions.rand(rng, spl)
    end
    partialsort!(s.buffer, s.k)
    s.buffer[s.k]
end

struct TukeyLambda <: Distribution{Univariate,Continuous}
    λ::Float64
end

function Statistics.quantile(d::TukeyLambda, q::Real)
    0 <= q <= 1 || throw(DomainError(q, "Expected 0 <= q <= 1."))
    if d.λ == 0
        return log(q / (1-q))
    else
        return (q^d.λ - (1-q)^d.λ) / d.λ
    end
end