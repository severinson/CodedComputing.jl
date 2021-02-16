using Distributions
export ExponentialOrder, OrderStatistic

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

struct OrderStatistic{S<:Sampleable{Univariate,Continuous},T} <: Sampleable{Univariate,Continuous}
    s::S
    k::Int
    buffer::Vector{T}
end

Base.show(io::IO, s::OrderStatistic) = print(io, "OrderStatistic($(s.s), k=$(s.k), n=$(length(s.buffer)))")

function OrderStatistic(s::Sampleable, order::Integer, nvariables::Integer)
    OrderStatistic(s, order, Vector{eltype(s)}(undef, nvariables))
end

function Distributions.rand(rng::AbstractRNG, s::OrderStatistic)
    Distributions.rand!(rng, s.s, s.buffer)
    partialsort!(s.buffer, s.k)
    s.buffer[s.k]
end