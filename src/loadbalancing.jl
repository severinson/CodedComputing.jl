# Code for load-balancing optimization

using Random, Distributions

function setup_loadbalancer_channels(;chin_size=Inf, chout_size=Inf)
    chin = Channel{ProfilerOutput}(chin_size)
    chout = Channel{@NamedTuple{worker::Int,p::Int}}(chout_size)
    chin, chout
end

"""

For each random variable, return a lower bound on the log of the probability of a sample drawn 
from this variable being smaller than all other variables.
"""
function less_than_lower_bound!(rv, dzs, dys)
    length(rv) == length(dzs) || throw(DimensionMismatch("rv has dimension $(length(rv)), but dzs has dimension $(length(dzs))"))
    length(rv) == length(dys) || throw(DimensionMismatch("rv has dimension $(length(rv)), but dys has dimension $(length(dys))"))
    n = length(rv)
    
    # compute the midpoint of the means
    cz = mean(mean, dzs)
    cy = mean(mean, dys)    

    # for each i, prob. of (zi <= cz and all others >= cz) and (yi <= cy and all others >= cy)
    pz, py = 0.0, 0.0
    for i in 1:n
        pz += logccdf(dzs[i], cz)
        py += logccdf(dys[i], cy)        
    end
    for i in 1:n
        vz = logccdf(dzs[i], cz)
        vy = logccdf(dys[i], cy)        
        pz -= vz
        py -= vy
        rv[i] = max(rv[i], pz + logcdf(dzs[i], cz) + py + logcdf(dys[i], cy))
        pz += vz
        py += vy
    end
    rv
end

less_than_lower_bound(dzs, dys) = less_than_lower_bound!(fill(-Inf, length(dzs)), dzs, dys)

"""

Setup the simulator and simulate the fraction of iterations each worker participates in.
"""
function simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    nworkers = length(ls)

    # setup compute latency distributions
    for i in 1:nworkers
        m = comp_mcs[i] * θs[i] / ps[i]
        v = comp_vcs[i] * θs[i] / ps[i]
        sim.comp_distributions[i] = CodedComputing.distribution_from_mean_variance(Gamma, m, v)
    end

    # run nsamples simulations, each consisting of niterations steps
    ls .= 0
    for _ in 1:simulation_nsamples
        empty!(sim)
        step!(sim, simulation_niterations)
        ls .+= sim.nfresh ./ simulation_niterations
    end
    ls ./= simulation_nsamples
    ls .= log.(ls)
    less_than_lower_bound!(ls, sim.comp_distributions, sim.comm_distributions)
    ls
end

# estimate the gradient of the i-th element
function finite_diff(ps::AbstractVector, i::Integer, δ::Real=min(ps[i]-sqrt(eps(Float64)), ps[i]/10); ls, sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    0 < δ < ps[i] || throw(ArgumentError("δ is $δ, ps[i] is $(ps[i])"))
    0 < i <= length(ps) || throw(ArgumentError("i is $i"))
    pi0 = ps[i]

    # forward difference
    ps[i] = pi0 + δ
    ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    forward = ls[i] + log(θs[i]) - log(ps[i])

    # backward difference
    ps[i] = pi0 - δ
    ps[i] = max(sqrt(eps(Float64)), ps[i])
    ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    backward = ls[i] + log(θs[i]) - log(ps[i])

    # symmetric difference
    h = pi0 + δ - ps[i]
    ps[i] = pi0
    (forward - backward) / h
end

function optimize!(ps::AbstractVector, ps_prev::AbstractVector, sim::EventDrivenSimulator; ∇s=zeros(length(ps)), ls=zeros(length(ps)), contribs=zeros(length(ps)), θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_processed_fraction::Real, time_limit::Real=1.0, simulation_niterations::Integer=100, simulation_nsamples::Integer=10, aggressive::Bool=false, min_latency::Float64=0.0)
    0 < min_processed_fraction <= 1 || throw(ArgumentError("min_processed_fraction is $min_processed_fraction"))
    nworkers = length(ps)
    length(ps_prev) == nworkers || throw(DimensionMismatch("ps_prev has dimension $(length(ps_prev)), but nworkers is $nworkers"))    
    length(θs) == nworkers || throw(DimensionMismatch("θs has dimension $(length(θs)), but nworkers is $nworkers"))    
    length(comp_mcs) == nworkers || throw(DimensionMismatch("comp_mcs has dimension $(length(comp_mcs)), but nworkers is $nworkers"))
    length(comp_vcs) == nworkers || throw(DimensionMismatch("comp_vcs has dimension $(length(comp_vcs)), but nworkers is $nworkers"))
    length(comm_mcs) == nworkers || throw(DimensionMismatch("comm_mcs has dimension $(length(comm_mcs)), but nworkers is $nworkers"))
    length(comm_vcs) == nworkers || throw(DimensionMismatch("comm_vcs has dimension $(length(comm_vcs)), but nworkers is $nworkers"))

    # negative min_latency indicates that it should be set to the minimum latency over all workers
    if min_latency < 0
        min_latency = comp_mcs[1] * θs[1] / ps[1] + comm_mcs[1]
        for i in 2:nworkers            
            min_latency = min(min_latency, comp_mcs[i] * θs[i] / ps[i] + comm_mcs[i])
        end
    end

    # setup communication latency distributions
    for i in 1:nworkers
        m = comm_mcs[i]
        v = comm_vcs[i]
        sim.comm_distributions[i] = CodedComputing.distribution_from_mean_variance(Gamma, m, v)
    end

    # parameters
    μ = min_processed_fraction / nworkers
    α = 0.1    
    β = 0.1

    # loss for previous solution for reference
    ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    contribs .= ls .+ log.(θs) .- log.(ps_prev)
    loss0 = maximum(contribs) - minimum(contribs)

    # initialization
    # scale the workload uniformly to meet the min_processed_fraction requirement
    if !aggressive
        contribs .= θs ./ ps
        ps ./= (sim.nworkers / sim.nwait) * min_processed_fraction / sum(contribs)
    end

    # compute per-worker contributions
    ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    contribs .= ls .+ log.(θs) .- log.(ps)

    # run for up to time_limit seconds
    t0 = time_ns()
    t = t0
    while (t - t0) / 1e9 < time_limit && t0 <= t

        # increase contribution of worker with smallest contribution
        i = argmin(contribs)
        ∇ = finite_diff(ps, i; ls, sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        x = (log(μ) - contribs[i]) / ∇ * β + ps[i]
        x = min(x, (1+α)*ps[i])
        x = max(x, (1-α)*ps[i])
        ps[i] = x
        if comm_mcs[i] <= min_latency
            ps[i] = min(ps[i], comp_mcs[i] * θs[i] / (min_latency - comm_mcs[i]))
        end

        # scale the workload uniformly to meet the min_processed_fraction requirement
        if !aggressive
            contribs .= θs ./ ps
            ps ./= (sim.nworkers / sim.nwait) * min_processed_fraction / sum(contribs)
        end

        # compute per-worker contributions
        ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        contribs .= ls .+ log.(θs) .- log.(ps)

        t = time_ns()
    end

    loss = maximum(contribs) - minimum(contribs)
    ps, loss, loss0
end

function load_balancer(chin::Channel, chout::Channel; min_processed_fraction::Real, nwait::Integer, nworkers::Integer, nsubpartitions::Union{Integer,<:AbstractVector}, time_limit::Real=1.0, min_improvement::Real=10)
    0 < min_processed_fraction <= 1 || throw(ArgumentError("min_processed_fraction is $min_processed_fraction"))
    0 < nworkers || throw(ArgumentError("nworkers is $nworkers"))
    0 < nwait <= nworkers || throw(ArgumentError("nwait is $nwait, but nworkers is $nworkers"))
    if typeof(nsubpartitions) <: Integer
        0 < nsubpartitions || throw(ArgumentError("nsubpartitions must be positive, but is $nsubpartitions"))
    else
        length(nsubpartitions) == nworkers || throw(DimensionMismatch("nsubpartitions has dimension $(length(nsubpartitions)), but nworkers is $nworkers"))
    end
    @info "load_balancer task started on thread $(Threads.threadid())"

    # fraction of dataset stored by and fraction of local data processed per iteration for each worker
    θs = fill(1/nworkers, nworkers)
    ps = typeof(nsubpartitions) <: Integer ? fill(float(nsubpartitions), nworkers) : float.(nsubpartitions)
    ps_prev = copy(ps)

    # mean and variance coefficients for each worker
    comp_mcs = fill(NaN, nworkers)
    comp_vcs = fill(NaN, nworkers)
    comm_mcs = fill(NaN, nworkers)
    comm_vcs = fill(NaN, nworkers)

    # buffers used by the optimizer
    ls = zeros(nworkers)
    contribs = zeros(nworkers)
    ∇s = zeros(nworkers)

    # reusable simulator
    comp_distributions = [Gamma() for _ in 1:nworkers]
    comm_distributions = [Gamma() for _ in 1:nworkers]
    sim = EventDrivenSimulator(;nwait, nworkers, comp_distributions, comm_distributions)

    # helper to check if there is any missing latency data
    # (in which case we shouldn't run the load-balancer)
    all_populated = false
    function check_populated()
        rv = iszero(count(isnan, comp_mcs))
        if rv
            rv = rv && iszero(count(isnan, comp_vcs))
        end
        if rv
            rv = rv && iszero(count(isnan, comm_mcs))
        end
        if rv
            rv = rv && iszero(count(isnan, comm_vcs))
        end
        if rv
            rv = rv && iszero(count(isnan, θs))
        end
        if rv
            rv = rv && iszero(count(isnan, ps))
        end
        rv
    end

    # consume incoming ProfilerOutput samples and run the optimizer
    while isopen(chin)

        # consume all values currently in the channel
        try
            vin = take!(chin)
            isnan(vin.comp_mc) || (comp_mcs[vin.worker] = vin.comp_mc)
            isnan(vin.comp_vc) || (comp_vcs[vin.worker] = vin.comp_vc)
            isnan(vin.comm_mc) || (comm_mcs[vin.worker] = vin.comm_mc)
            isnan(vin.comm_vc) || (comm_vcs[vin.worker] = vin.comm_vc)
        catch e
            if e isa InvalidStateException
                @info "error taking value from input channel" e          
                break
            else
                rethrow()
            end
        end            
        while isready(chin)
            try
                vin = take!(chin)
                isnan(vin.comp_mc) || (comp_mcs[vin.worker] = vin.comp_mc)
                isnan(vin.comp_vc) || (comp_vcs[vin.worker] = vin.comp_vc)
                isnan(vin.comm_mc) || (comm_mcs[vin.worker] = vin.comm_mc)
                isnan(vin.comm_vc) || (comm_vcs[vin.worker] = vin.comm_vc)                
            catch e
                if e isa InvalidStateException
                    @info "error taking value from input channel" e          
                    break
                else
                    rethrow()
                end
            end
        end
    
        # verify that we have complete latency information for all workers
        if !all_populated
            all_populated = check_populated()
        end
        if !all_populated
            continue
        end

        try
            # @info "running load-balancer w. ps: $ps, θs: $θs, comp_mcs: $comp_mcs, comp_vcs: $comp_vcs"
            t = @timed begin
                ps, loss, loss0 = optimize!(ps, ps_prev, sim; ∇s, ls, contribs, θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_processed_fraction, time_limit)
            end

            # compare the initial and new solutions, and continue if the change isn't large enough
            if isnan(loss) || isinf(loss) || loss0 / loss < min_improvement
                @info "load-balancer allocated $(t.bytes / 1e6) MB, and finished in $(t.time) seconds with loss $loss and loss0 $loss0; continuing"
                continue
            end
            @info "load-balancer allocated $(t.bytes / 1e6) MB, and finished in $(t.time) seconds with loss $loss and loss0 $loss0, a $(loss0/loss) fraction improvement"
            ps_prev .= ps

            # push any changes into the output channel
            for i in 1:nworkers
                p = max(1, round(Int, ps[i]))
                vout = @NamedTuple{worker::Int,p::Int}((i, p))
                try
                    push!(chout, vout)
                catch e
                    if e isa InvalidStateException
                        @info "error pushing value into output channel" e
                        break
                    else
                        rethrow()
                    end
                end
            end
        catch e
            if e isa ArgumentError
                @error "load-balancer failed; trying again" e                
            else
                rethrow()
            end
        end
    end
    @info "load_balancer task finished"
    return
end