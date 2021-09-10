# Code for running event-driven simulations to predict latency
export EventDrivenSimulator
export step!

function distribution_from_mean_variance(::Type{Gamma}, m, v)
    θ = v / m
    α = m / θ
    Gamma(α, θ)
end

mutable struct EventDrivenSimulator{Td1<:Sampleable{Univariate},Td2<:Sampleable{Univariate}}
    epoch::Int
    time::Float64
    nwait::Int
    nworkers::Int
    isidle::Vector{Bool}
    sepoch::Vector{Int}
    nfresh::Vector{Int}
    nstale::Vector{Int}
    comp_distributions::Vector{Td1}
    comm_distributions::Vector{Td2}
    pq::BinaryMinHeap{Tuple{Float64, Int}} # (time the worker becomes available at, worker index)
end

"""

Construct a new simulator from scratch.
"""
function EventDrivenSimulator(;nwait::Integer, nworkers::Integer, comm_distributions::AbstractVector{<:Sampleable{Univariate}}, comp_distributions::AbstractVector{<:Sampleable{Univariate}})
    EventDrivenSimulator(
        0, 0.0, nwait, nworkers, 
        ones(Bool, nworkers),
        zeros(Int, nworkers),
        zeros(Int, nworkers),
        zeros(Int, nworkers),
        comp_distributions,
        comm_distributions,
        sizehint!(BinaryMinHeap{Tuple{Float64,Int}}(), nworkers),
    )
end

"""

Construct a new simulator from a previous one, re-using the arrays already allocated for `sim`.
"""
function EventDrivenSimulator(sim::EventDrivenSimulator; nwait::Integer=sim.nwait, nworkers::Integer=sim.nworkers, 
    comm_distributions::AbstractVector{<:Sampleable{Univariate}}=sim.comm_distributions, 
    comp_distributions::AbstractVector{<:Sampleable{Univariate}}=sim.comp_distributions)
    sim.isidle .= true
    sim.sepoch .= 0
    sim.nfresh .= 0
    sim.nstale .= 0
    while length(sim.pq) > 0
        pop!(sim.pq)
    end
    EventDrivenSimulator(
        0, 0.0, nwait, nworkers,
        sim.isidle,
        sim.sepoch,
        sim.nfresh,
        sim.nstale,
        comp_distributions,
        comm_distributions,
        sim.pq,
    )
end

function DataStructures.enqueue!(sim::EventDrivenSimulator, i::Integer)
    comp, comm = rand(sim.comp_distributions[i]), rand(sim.comm_distributions[i])
    (0 <= comp && 0 <= comm) || throw(ArgumentError("delay must be positive, but is $((comp, comm))"))
    delay = comp + comm
    push!(sim.pq, (sim.time+delay, i))
    sim.isidle[i] = false # worker becomes busy
    sim.sepoch[i] = sim.epoch # register the epoch in which the task was assigned
    return
end

"""

Run one epoch of event-driven simulation.
"""
function step!(sim)

    # start a new simulation epoch
    sim.epoch += 1

    # enqueue all idle workers
    for i in 1:sim.nworkers
        if sim.isidle[i]
            enqueue!(sim, i)
        end
    end

    # wait until nwait workers have finished
    nfresh = 0
    while nfresh < sim.nwait
        t, i = pop!(sim.pq)
        sim.time = t
        sim.isidle[i] = true
        if sim.sepoch[i] == sim.epoch # result is fresh
            sim.nfresh[i] += 1
            nfresh += 1
        else # result is stale
            sim.nstale[i] += 1
            enqueue!(sim, i) # late workers are assigned a new task
        end
    end
    sim
end

function step!(sim, n::Integer)
    0 < n || throw(ArgumentError("n must be positive, but is $n"))
    for _ in 1:n
        step!(sim)
    end
end