# Code for load-balancing optimization

using Random, Distributions

function setup_loadbalancer_channels(;chin_size=200, chout_size=200)
    chin = ConcurrentCircularBuffer{ProfilerOutput}(chin_size)
    chout = ConcurrentCircularBuffer{@NamedTuple{worker::Int,p::Int}}(chout_size)
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
        v = comp_vcs[i] * (θs[i] / ps[i])^2
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

"""

Return the expected latency of the `i`-th worker.
"""
expected_worker_latency(i; θs, ps, comp_mcs, comm_mcs) = comp_mcs[i] * θs[i] / ps[i] + comm_mcs[i]

function min_max_expected_worker_latency(;θs, ps, comp_mcs, comm_mcs)
    nworkers = length(ps)
    vmin, vmax = Inf, -Inf
    for i in 1:nworkers
        v = expected_worker_latency(i; θs, ps, comp_mcs, comm_mcs)
        vmin = min(vmin, v)
        vmax = max(vmax, v)
    end
    vmin, vmax
end

function optimize!(ps::AbstractVector, ps_prev::AbstractVector, sim::EventDrivenSimulator; ls=zeros(length(ps)), contribs=zeros(length(ps)), θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_contribution::Real=(sim.nwait / length(ps_prev) / mean(ps_prev)), min_comp_fraction::Real=0.1, simulation_niterations::Integer=50, simulation_nsamples::Integer=3)
    nworkers = length(ps)
    0 < min_contribution <= 1 || throw(ArgumentError("min_contribution is $min_contribution"))
    length(ps_prev) == nworkers || throw(DimensionMismatch("ps_prev has dimension $(length(ps_prev)), but nworkers is $nworkers"))    
    length(θs) == nworkers || throw(DimensionMismatch("θs has dimension $(length(θs)), but nworkers is $nworkers"))    
    length(comp_mcs) == nworkers || throw(DimensionMismatch("comp_mcs has dimension $(length(comp_mcs)), but nworkers is $nworkers"))
    length(comp_vcs) == nworkers || throw(DimensionMismatch("comp_vcs has dimension $(length(comp_vcs)), but nworkers is $nworkers"))
    length(comm_mcs) == nworkers || throw(DimensionMismatch("comm_mcs has dimension $(length(comm_mcs)), but nworkers is $nworkers"))
    length(comm_vcs) == nworkers || throw(DimensionMismatch("comm_vcs has dimension $(length(comm_vcs)), but nworkers is $nworkers"))
    0 < min_comp_fraction < 1 || throw(ArgumentError("min_comp_fraction is $min_comp_fraction"))

    # setup communication latency distributions
    for i in 1:nworkers
        m = comm_mcs[i]
        v = comm_vcs[i]
        sim.comm_distributions[i] = CodedComputing.distribution_from_mean_variance(Gamma, m, v)
    end

    # loss for previous solution for reference
    latency0, ls = simulate!(ls, ps_prev; sim, θs, comp_mcs, comp_vcs, simulation_nsamples=2*simulation_nsamples, simulation_niterations=2*simulation_nsamples)
    contribs .= ls .+ log.(θs) .- log.(ps_prev)
    contrib0 = sum(exp, contribs)
    vmin, vmax = min_max_expected_worker_latency(;θs, ps=ps_prev, comp_mcs, comm_mcs)
    loss0 = vmax / vmin

    # equalize latency between workers
    ## find the slowest worker
    i = 0
    v = -Inf
    for j in 1:nworkers
        comp_latency = comp_mcs[j] * θs[j] / ps[j]
        if comp_latency + comm_mcs[j] > v
            i = j
            v = comp_latency + comm_mcs[j]
        end
    end

    ## increase the workload of all other workers, so that what was the slowest worker becomes the fastest
    for j in 1:nworkers
        if j == i
            continue
        end
        ps[j] = max(1.0, round(comp_mcs[j] * θs[j] / (v - comm_mcs[j])))
    end

    # latency and contribution of the current solution
    latency, ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples=2*simulation_nsamples, simulation_niterations=2*simulation_niterations)
    contribs .= ls .+ log.(θs) .- log.(ps)
    contrib = sum(exp, contribs)

    # while below the contribution constraint, assign more work to the fastest workers
    while contrib < min_contribution

        # find the fastest worker with at least 2 sub-partitions
        i = 0
        v = Inf
        for j in 1:nworkers
            comp_latency = comp_mcs[j] * θs[j] / ps[j]
            if ps[j] >= 2 && (comp_latency + comm_mcs[j]) < v
                i = j
                v = comp_latency + comm_mcs[j]
            end
        end
        if iszero(i)
            break
        end
        δ = max(1, round(ps[i] * 0.01))
        ps[i] -= δ

        latency, ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        contribs .= ls .+ log.(θs) .- log.(ps)
        contrib = sum(exp, contribs)

        # double check the exit condition
        if contrib >= min_contribution
            latency, ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples=2*simulation_nsamples, simulation_niterations=2*simulation_niterations)
            contribs .= ls .+ log.(θs) .- log.(ps)
            contrib = sum(exp, contribs)
        end
    end

    # while within the contribution constraint, reduce the workload of slow workers
    while contrib >= min_contribution * 0.99

        # find the slowest worker with a comp. latency that accounts for at least 
        # min_comp_fraction of the overall latency of the worker
        # (to ensure that the load-balancer doesn't push comp. latency to zero)
        i = 0
        v = -Inf
        for j in 1:nworkers
            comp_latency = comp_mcs[j] * θs[j] / ps[j]
            if comp_latency / (comp_latency + comm_mcs[j]) >= min_comp_fraction && (comp_latency + comm_mcs[j]) > v
                i = j
                v = comp_latency + comm_mcs[j]
            end
        end
        if iszero(i)
            break
        end
        δ = max(1, round(ps[i] * 0.01))
        ps[i] += δ
        latency, ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        contribs .= ls .+ log.(θs) .- log.(ps)
        contrib = sum(exp, contribs)

        # double check the exit condition
        if contrib < min_contribution * 0.99
            latency, ls = simulate!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples=2*simulation_nsamples, simulation_niterations=2*simulation_niterations)
            contribs .= ls .+ log.(θs) .- log.(ps)
            contrib = sum(exp, contribs)
        end
    end

    vmin, vmax = min_max_expected_worker_latency(;θs, ps, comp_mcs, comm_mcs)
    loss = vmax / vmin

    ps, latency0, contrib0, loss0, latency, contrib, loss
end

function load_balancer(chin::ConcurrentCircularBuffer, chout::ConcurrentCircularBuffer; nwait::Integer, nworkers::Integer, nsubpartitions::Union{Integer,<:AbstractVector}, min_improvement::Real=0.9)
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

    # minimum contribution set to that of the initial solution
    min_contribution = nwait / length(ps_prev) / mean(ps_prev)

    # mean and variance coefficients for each worker
    # (dummy values used to force pre-compilation of the optimizer)
    comp_mcs = ones(nworkers)
    comp_vcs = ones(nworkers)
    comm_mcs = ones(nworkers)
    comm_vcs = ones(nworkers)

    # buffers used by the optimizer
    ls = zeros(nworkers)
    contribs = zeros(nworkers)

    # reusable simulator
    comp_distributions = [Gamma() for _ in 1:nworkers]
    comm_distributions = [Gamma() for _ in 1:nworkers]
    sim = EventDrivenSimulator(;nwait, nworkers, comp_distributions, comm_distributions)

    # run the optimizer once to force pre-compilation
    optimize!(ps, ps_prev, sim; ls, contribs, θs, min_contribution, comp_mcs, comp_vcs, comm_mcs, comm_vcs)

    # reset changes made by the optimizer
    ps .= nsubpartitions
    ps_prev .= nsubpartitions
    comp_mcs .= NaN
    comp_vcs .= NaN
    comm_mcs .= NaN
    comm_vcs .= NaN

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
            if all_populated
                @info "load_balancer has latency information for all workers; starting to optimize"
            end
        end
        if !all_populated
            continue
        end

        try
            # @info "load_balancer optimization started with ps: $ps, θs: $θs, comp_mcs: $comp_mcs, comp_vcs: $comp_vcs, comm_mcs: $comm_mcs, comm_vcs: $comm_vcs"
            # @info "load_balancer optimization started with ps = $ps, θs = $θs, comp_mcs = $comp_mcs, comp_vcs = $comp_vcs, comm_mcs = $comm_mcs, comm_vcs = $comm_vcs"
            # @info "ps_prev: $ps_prev, ps_baseline: $ps_baseline"
            # @info "load_balancer optimization started"
            t = @elapsed begin
                ps, latency0, contrib0, loss0, latency, contrib, loss = optimize!(ps, ps_prev, sim; ls, contribs, θs, min_contribution, comp_mcs, comp_vcs, comm_mcs, comm_vcs)
            end

            # compare the initial and new solutions, and continue if the change isn't large enough
            if isnan(loss) || isinf(loss) || (loss / loss0) > min_improvement
                # @info "load-balancer finished in $(t) seconds with loss $loss and loss0 $loss0; continuing. ps: $ps"
                continue
            end
            # @info "load-balancer finished in $(t) seconds with loss $loss and loss0 $loss0; accepting it. ps: $ps"
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