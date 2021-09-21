# Code for load-balancing optimization

using Random, Distributions

function setup_loadbalancer_channels(;chin_size=Inf, chout_size=Inf)
    chin = Channel{ProfilerOutput}(chin_size)
    chout = Channel{@NamedTuple{worker::Int,p::Int}}(chout_size)
    chin, chout
end

"""

For each worker, return a lower bound on the probability of this worker being the fastest of all
workers, i.e., when `nwait < nworkers`, `rv[i]` is a lower bound on `log(ls[i])`.
"""
function fastest_lower_bound!(rv, dzs, dys)
    length(rv) == length(dzs) || throw(DimensionMismatch("rv has dimension $(length(rv)), but dzs has dimension $(length(dzs))"))
    length(rv) == length(dys) || throw(DimensionMismatch("rv has dimension $(length(rv)), but dys has dimension $(length(dys))"))
    n = length(rv)
    
    # compute the midpoint of the means
    cz = mean(mean, dzs)
    cy = mean(mean, dys)

    # for each i, prob. of (zi <= cz and all others > cz) and (yi <= cy and all others > cy)
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

fastest_lower_bound(dzs, dys) = fastest_lower_bound!(fill(-Inf, length(dzs)), dzs, dys)

"""

For each worker, return a lower bound on the probability of this worker being the slowest of all
workers.
"""
function slowest_lower_bound!(rv, dzs, dys)
    length(rv) == length(dzs) || throw(DimensionMismatch("rv has dimension $(length(rv)), but dzs has dimension $(length(dzs))"))
    length(rv) == length(dys) || throw(DimensionMismatch("rv has dimension $(length(rv)), but dys has dimension $(length(dys))"))
    n = length(rv)

    # compute the midpoint of the means
    cz = mean(mean, dzs)
    cy = mean(mean, dys)    

    # for each i, prob. of (zi > cz and all others <= cz) and (yi > cy and all others <= cy)
    pz, py = 0.0, 0.0
    for i in 1:n
        pz += logcdf(dzs[i], cz)
        py += logcdf(dys[i], cy)
    end
    for i in 1:n
        vz = logcdf(dzs[i], cz)
        vy = logcdf(dys[i], cy)        
        pz -= vz
        py -= vy
        rv[i] = min(rv[i], pz + logccdf(dzs[i], cz) + py + logccdf(dys[i], cy))
        pz += vz
        py += vy
    end
    rv
end

slowest_lower_bound(dzs, dys) = slowest_lower_bound!(fill(Inf, length(dzs)), dzs, dys)

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
    latency = 0.0
    ls .= 0
    for _ in 1:simulation_nsamples
        empty!(sim)
        step!(sim, simulation_niterations)
        latency += sim.time / simulation_niterations
        ls .+= sim.nfresh ./ simulation_niterations
    end
    latency /= simulation_nsamples
    ls ./= simulation_nsamples
    ls .= log.(ls)
    fastest_lower_bound!(ls, sim.comp_distributions, sim.comm_distributions)
    latency, ls
end

function optimize!(ps::AbstractVector, ps_prev::AbstractVector, sim::EventDrivenSimulator; ps_baseline=ps_prev, ls=zeros(length(ps)), contribs=zeros(length(ps)), θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, simulation_niterations::Integer=100, simulation_nsamples::Integer=10)
    nworkers = length(ps)
    length(ps_prev) == nworkers || throw(DimensionMismatch("ps_prev has dimension $(length(ps_prev)), but nworkers is $nworkers"))    
    length(ps_baseline) == nworkers || throw(DimensionMismatch("ps_baseline has dimension $(length(ps_baseline)), but nworkers is $nworkers"))    
    length(θs) == nworkers || throw(DimensionMismatch("θs has dimension $(length(θs)), but nworkers is $nworkers"))    
    length(comp_mcs) == nworkers || throw(DimensionMismatch("comp_mcs has dimension $(length(comp_mcs)), but nworkers is $nworkers"))
    length(comp_vcs) == nworkers || throw(DimensionMismatch("comp_vcs has dimension $(length(comp_vcs)), but nworkers is $nworkers"))
    length(comm_mcs) == nworkers || throw(DimensionMismatch("comm_mcs has dimension $(length(comm_mcs)), but nworkers is $nworkers"))
    length(comm_vcs) == nworkers || throw(DimensionMismatch("comm_vcs has dimension $(length(comm_vcs)), but nworkers is $nworkers"))

    # setup communication latency distributions
    for i in 1:nworkers
        m = comm_mcs[i]
        v = comm_vcs[i]
        sim.comm_distributions[i] = CodedComputing.distribution_from_mean_variance(Gamma, m, v)
    end

    # loss for previous solution for reference
    latency0, ls = simulate!(ls, ps_prev; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    contribs .= ls .+ log.(θs) .- log.(ps_prev)
    contrib0 = sum(exp, contribs)
    loss0 = maximum(ls) - minimum(ls)

    # check if the baseline and prev vectors are the same
    baseline_is_prev = true
    for i in 1:nworkers
        if ps_prev[i] != ps_baseline[i]
            baseline_is_prev = false
            break
        end
    end

    # compute loss and latency constraints from the baseline
    min_contribution = contrib0
    max_latency = latency0
    if !baseline_is_prev 
        max_latency, ls = simulate!(ls, ps_baseline; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        contribs .= ls .+ log.(θs) .- log.(ps_prev)
        min_contribution = sum(exp, contribs)
    end

    # latency and contribution of the current solution
    latency, ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    contribs .= ls .+ log.(θs) .- log.(ps)
    contrib = sum(exp, contribs)

    # while within the latency constraint, slow down the fastest workers
    i = 0
    while isapprox(latency, max_latency, rtol=1e-2) || latency < max_latency

        # find the fastest worker with at least 2 sub-partitions
        i = 0
        v = -Inf
        for j in 1:nworkers
            if ps[j] >= 2 && ls[j] > v
                i = j
                v = ls[j]
            end
        end
        @assert i != 0
        ps[i] -= 1
        latency, ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        contribs .= ls .+ log.(θs) .- log.(ps)
        contrib = sum(exp, contribs)
    end
    if !iszero(i)
        ps[i] += 1
        latency, ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        contribs .= ls .+ log.(θs) .- log.(ps)
        contrib = sum(exp, contribs)        
    end

    # while within the contribution constraint, speed up the slowest workers
    i = 0
    while isapprox(contrib, min_contribution, rtol=1e-2) || min_contribution < contrib        

        # find the slowest worker with a comm. latency less than the maximum latency
        # (since other workers can't be load-balanced)
        i = 0
        v = Inf
        for j in 1:nworkers
            if comm_mcs[j] < max_latency * 0.9 && ls[j] < v
                i = j
                v = ls[j]
            end
        end
        @assert i != 0
        ps[i] += 1
        latency, ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        contribs .= ls .+ log.(θs) .- log.(ps)
        contrib = sum(exp, contribs)        
    end
    if !iszero(i)
        ps[i] -= 1
        latency, ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        contribs .= ls .+ log.(θs) .- log.(ps)
        contrib = sum(exp, contribs)        
    end
    loss = maximum(ls) - minimum(ls)
    ps, latency0, contrib0, loss0, latency, contrib, loss
end

function load_balancer(chin::Channel, chout::Channel; min_processed_fraction::Real, nwait::Integer, nworkers::Integer, nsubpartitions::Union{Integer,<:AbstractVector}, time_limit::Real=1.0, min_improvement::Real=0.9, aggressive::Bool=false)
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
    ps_baseline = copy(ps)

    # mean and variance coefficients for each worker
    comp_mcs = fill(NaN, nworkers)
    comp_vcs = fill(NaN, nworkers)
    comm_mcs = fill(NaN, nworkers)
    comm_vcs = fill(NaN, nworkers)

    # buffers used by the optimizer
    ls = zeros(nworkers)
    contribs = zeros(nworkers)

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
                ps, latency0, contrib0, loss0, latency, contrib, loss = optimize!(ps, ps_prev, sim; ps_baseline, ls, contribs, θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs)
            end

            # compare the initial and new solutions, and continue if the change isn't large enough
            if isnan(loss) || isinf(loss) || exp(loss - loss0) < min_improvement
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