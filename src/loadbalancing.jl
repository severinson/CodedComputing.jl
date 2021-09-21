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
    ls .= 0
    for _ in 1:simulation_nsamples
        empty!(sim)
        step!(sim, simulation_niterations)
        ls .+= sim.nfresh ./ simulation_niterations
    end
    ls ./= simulation_nsamples
    ls .= log.(ls)
    fastest_lower_bound!(ls, sim.comp_distributions, sim.comm_distributions)
    ls
end

"""

Setup the simulator and simulate the fraction of iterations each worker participates in.
"""
function simulate2!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
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

# estimate the gradient of the i-th element
function finite_diff2(ps::AbstractVector, i::Integer, δ::Real=min(ps[i]-sqrt(eps(Float64)), ps[i]/10); ls, sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    0 < δ < ps[i] || throw(ArgumentError("δ is $δ, ps[i] is $(ps[i])"))
    0 < i <= length(ps) || throw(ArgumentError("i is $i"))
    pi0 = ps[i]

    # forward difference
    ps[i] = pi0 + δ
    forward_latency, ls = simulate2!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    
    ls .= exp.(ls)
    ls .*= θs ./ ps        
    
    # ls .+= log.(θs) .- log.(ps)
    
    forward_contrib = sum(ls)

    # backward difference
    ps[i] = pi0 - δ
    ps[i] = max(sqrt(eps(Float64)), ps[i])
    backward_latency, ls = simulate2!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)

    ls .= exp.(ls)
    ls .*= θs ./ ps            

    # ls .+= log.(θs) .- log.(ps)
    
    backward_contrib = sum(ls)

    # symmetric difference
    h = pi0 + δ - ps[i]
    ps[i] = pi0
    (forward_latency - backward_latency) / h, (forward_contrib - backward_contrib) / h
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

function optimize2!(ps::AbstractVector, ps_prev::AbstractVector, sim::EventDrivenSimulator; ls=zeros(length(ps)), contribs=zeros(length(ps)), is=collect(1:length(ps)), θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, time_limit::Real=1.0, simulation_niterations::Integer=100, simulation_nsamples::Integer=10, min_contribution::Float64=NaN, max_latency::Float64=NaN)
    0 < min_contribution <= 1 || throw(ArgumentError("min_contribution is $min_contribution"))
    nworkers = length(ps)
    length(ps_prev) == nworkers || throw(DimensionMismatch("ps_prev has dimension $(length(ps_prev)), but nworkers is $nworkers"))    
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

    # compute the minimum latency across all workers
    # (used later to check if it's possible to optimize a worker)
    min_latency = Inf
    for i in 1:nworkers
        min_latency = min(min_latency, comm_mcs[i] + comp_mcs[i] * θs[i] / ps[i])
    end

    # loss for previous solution for reference
    latency0, ls = simulate2!(ls, ps_prev; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    contribs .= ls .+ log.(θs) .- log.(ps_prev)
    contrib0 = sum(exp, contribs)

    # max_latency = NaN indicates it should be set to that of the initial solution
    if isnan(max_latency)
        max_latency = latency0
    end

    # min_contribution = NaN indicates it should be set to that of the initial solution
    if isnan(min_contribution)
        min_contribution = contrib0
    end

    # baseline latency and contribution for comparison with when evaluating new solutions    
    latency, ls = simulate2!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    contribs .= ls .+ log.(θs) .- log.(ps)
    contrib = sum(exp, contribs)

    # run-to-convergence algorithm
    # # δ = 0
    # # while within the latency constraint, slow down the fastest workers
    # i = 0
    # while isapprox(latency, max_latency, rtol=1e-2) || latency < max_latency
    #     i = argmax(ls)
    #     # δ = -1
    #     ps[i] -= 1

    #     latency, ls = simulate2!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    #     contribs .= ls .+ log.(θs) .- log.(ps)
    #     contrib = sum(exp, contribs)
    # end
    # if !iszero(i)
    #     @info "backing out -1"
    #     ps[i] += 1
    # end

    # latency, ls = simulate2!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    # contribs .= ls .+ log.(θs) .- log.(ps)
    # contrib = sum(exp, contribs)        

    # # while within the contribution constraint, speed up the slowest workers
    # # δ = 0
    # i = 0
    # while isapprox(contrib, min_contribution, rtol=1e-2) || min_contribution < contrib        
    #     i = argmin(ls)
    #     # δ = 1
    #     ps[i] += 1

    #     latency, ls = simulate2!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    #     contribs .= ls .+ log.(θs) .- log.(ps)
    #     contrib = sum(exp, contribs)
    # end
    # if !iszero(i)
    #     @info "backing out +1"
    #     ps[i] -= 1
    # end

    latency, ls = simulate2!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
    contribs .= ls .+ log.(θs) .- log.(ps)
    contrib = sum(exp, contribs)    

   

    return ps, latency0, contrib0, latency, contrib


    # rtol = 1e-2 / nworkers
    rtol = 1e-2

    # run for up to time_limit seconds
    t0 = time_ns()
    t = t0
    while (t - t0) / 1e9 < time_limit && t0 <= t

        # @info "min_contribution: $min_contribution, baseline_contrib: $baseline_contrib"

        δ = 0        
        if isapprox(contrib, min_contribution, rtol=1e-2) || min_contribution < contrib
            # reduce workload of the slowest worker
            i = argmin(ls)
            δ = 1
            # ps[i] += 1
        else
            # add work to the fastest worker
            i = argmax(ls)
            δ = -1
            # ps[i] -= 1            
        end
        ps[i] += δ

        latency, ls = simulate2!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        contribs .= ls .+ log.(θs) .- log.(ps)
        contrib = sum(exp, contribs)               
        if !isapprox(latency, max_latency, rtol=1e-2) && latency > max_latency # never increase latency by more than 1%
            ps[i] -= δ
            @info "backing out due to exceeding max_latency"
            break
        elseif !isapprox(contrib, min_contribution, rtol=1e-2) && contrib < min_contribution
            ps[i] -= δ
            @info "backing out due to exceeding min_contribution"
            break            
        end

        t = time_ns()
        continue

        # shuffle!(is)
        # for i in 1:nworkers

        #     println()

        #     # skip worker with very high communication latency
        #     # (even at zero workload the prob. of these workers participating is negligible)
        #     if comm_mcs[i] > min_latency
        #         @info "comm_mcs[$i] = $(comm_mcs[i]), min_latency = $min_latency; continuing"
        #         continue
        #     end

        #     # baseline number of partitions
        #     pi0 = ps[i]

        #     # try increasing the number of partitions
        #     δ = ps[i] / 100
        #     ps[i] = ceil(pi0 + δ)
        #     latency, ls = simulate2!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        #     ls .= exp.(ls)
        #     contribs .= ls .* θs ./ ps
        #     contrib = sum(contribs)
        #     @info "Increased ps[$i] to $(ps[i]), contrib: $(contrib / baseline_contrib) ($(contrib / min_contribution)), latency: $(latency / baseline_latency)"

        #     # always accept the new solution if the prob. of participation is 0, since the only 
        #     # way to increase participation is to reduce latency
        #     if isapprox(contribs[i], 0, atol=eps(Float64))
        #         baseline_latency = latency
        #         baseline_contrib = contrib
        #         @info "Zero participation prob; increasing ps[$i]"
        #         continue
        #     end

        #     # accept the solution if the contribution is still high enough and latency went down
        #     # if (isapprox(contrib, min_contribution; rtol) || min_contribution < contrib) && (!isapprox(latency, baseline_latency; rtol) && latency < baseline_latency)
        #     if (isapprox(contrib, min_contribution; rtol=1e-2) || min_contribution < contrib) && (latency < baseline_latency)
        #         baseline_latency = latency
        #         baseline_contrib = contrib
        #         @info "Sufficient contrib. and reduced latency; increasing ps[$i]"
        #         continue
        #     end
            
        #     # try decreasing the number of partitions
        #     δ = min(ps[i] - 1, ps[i] / 100)
        #     ps[i] = floor(pi0 - δ)
        #     latency, ls = simulate2!(ls, ps; sim, θs, comp_mcs, comp_vcs, simulation_nsamples, simulation_niterations)
        #     ls .= exp.(ls)
        #     contribs .= ls .* θs ./ ps
        #     contrib = sum(contribs)
        #     @info "Decreased ps[$i] to $(ps[i]), contrib: $(contrib / baseline_contrib) ($(contrib / min_contribution)), latency: $(latency / baseline_latency)"

        #     # accept if the contrib. increased, and is 1% or more below target, even if latency increases
        #     # if (!isapprox(contrib, baseline_contrib; rtol=1e-2) && baseline_contrib < contrib) && (!isapprox(contrib, min_contribution; rtol) && contrib < min_contribution)
        #     # if (!isapprox(contrib, baseline_contrib; rtol=1e-2) && baseline_contrib < contrib) && (contrib < min_contribution)
        #     if (baseline_contrib < contrib) && (!isapprox(contrib, min_contribution; rtol=1e-2) || contrib < min_contribution)
        #         baseline_latency = latency
        #         baseline_contrib = contrib
        #         @info "Too low contrib; decreasing ps[$i]"
        #         continue
        #     end

        #     # accept if contrib. increased
        #     println((!isapprox(contrib, baseline_contrib; rtol), baseline_contrib < contrib, isapprox(latency, baseline_latency; rtol), latency < baseline_latency))
        #     # if (!isapprox(contrib, baseline_contrib; rtol=1e-2) && baseline_contrib < contrib) && (isapprox(latency, baseline_latency; rtol) || latency < baseline_latency)
        #     # if (baseline_contrib < contrib) && (isapprox(latency, baseline_latency; rtol) || latency < baseline_latency)
        #     if (baseline_contrib < contrib)
        #         baseline_latency = latency
        #         baseline_contrib = contrib
        #         @info "Contrib increased; decreasing ps[$i]"
        #         continue
        #     end
            
        #     # if we haven't accepted any solution, reset and try to optimize another worker
        #     ps[i] = pi0
        #     @info "No solution accepted for worker $i; resetting"
        # end
        # t = time_ns()
    end

    @info "ls: $(exp.(ls))"

    ps, latency0, contrib0, baseline_latency, baseline_contrib
end


function optimize!(ps::AbstractVector, ps_prev::AbstractVector, sim::EventDrivenSimulator; ls=zeros(length(ps)), contribs=zeros(length(ps)), θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_processed_fraction::Real, time_limit::Real=1.0, simulation_niterations::Integer=100, simulation_nsamples::Integer=10, aggressive::Bool=false, min_latency::Float64=0.0)
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

function load_balancer(chin::Channel, chout::Channel; min_processed_fraction::Real, nwait::Integer, nworkers::Integer, nsubpartitions::Union{Integer,<:AbstractVector}, time_limit::Real=1.0, min_improvement::Real=10, aggressive::Bool=false)
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

    # never reduce worker latency to less than this
    # (set automatically after receiving latency values for all workers)
    min_latency = NaN

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

        # set min_latency to the latency of the fastest worker
        if isnan(min_latency)
            min_latency = comp_mcs[1] * θs[1] / ps[1] + comm_mcs[1]
            for i in 2:nworkers
                min_latency = min(min_latency, comp_mcs[i] * θs[i] / ps[i] + comm_mcs[i])
            end
        end

        try
            # @info "running load-balancer w. ps: $ps, θs: $θs, comp_mcs: $comp_mcs, comp_vcs: $comp_vcs"
            t = @timed begin
                ps, loss, loss0 = optimize!(ps, ps_prev, sim; ls, contribs, θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_processed_fraction, time_limit, min_latency, aggressive)
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