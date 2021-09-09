# Code for load-balancing optimization

using Random, Distributions, Evolutionary

function setup_loadbalancer_channels(;chin_size=Inf, chout_size=Inf)
    chin = Channel{ProfilerOutput}(chin_size)
    chout = Channel{@NamedTuple{worker::Int,p::Int}}(chout_size)
    chin, chout
end

function optimize2!(ps::AbstractVector, sim::EventDrivenSimulator; ∇s=zeros(length(ps)), ls=zeros(length(ps)), contribs=zeros(length(ps)), θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_processed_fraction::Real, time_limit::Real=1.0, simulation_niterations::Integer=100, simulation_nsamples::Integer=10)
    0 < min_processed_fraction <= 1 || throw(ArgumentError("min_processed_fraction is $min_processed_fraction"))
    nworkers = length(ps)
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

    # helper function to run simulations
    function simulate(ps)

        # setup compute latency distributions
        for i in 1:nworkers
            m = comp_mcs[i] * θs[i] / ps[i]
            v = comp_vcs[i] * θs[i] / ps[i]
            sim.comp_distributions[i] = CodedComputing.distribution_from_mean_variance(Gamma, m, v)
        end
    
        # run nsamples simulations, each consisting of niterations steps
        ls .= 0
        for _ in 1:simulation_nsamples
            sim = EventDrivenSimulator(sim)
            step!(sim, simulation_niterations)
            ls .+= sim.nfresh ./ simulation_niterations
        end
        ls ./= simulation_nsamples
    end

    # estimate the gradient of the i-th element
    function finite_diff(ps::AbstractVector, i::Integer, δ::Real=min(ps[i]-sqrt(eps(Float64)), ps[i]/10))
        0 < δ < ps[i] || throw(ArgumentError("δ is $δ, ps[i] is $(ps[i])"))
        0 < i <= length(ps) || throw(ArgumentError("i is $i"))
        pi0 = ps[i]

        # forward difference
        ps[i] = pi0 + δ
        ls = simulate(ps)
        forward = ls[i] * θs[i] / ps[i]

        # backward difference
        ps[i] = pi0 - δ
        ls = simulate(ps)
        backward = ls[i] * θs[i] / ps[i]

        # symmetric difference
        ps[i] = pi0
        (forward - backward) / 2δ
    end

    # parameters
    μ = min_processed_fraction / nworkers
    α = 0.1
    
    # run for up to time_limit seconds
    t0 = time_ns()
    t = t0    
    while (t - t0) / 1e9 < time_limit && t0 <= t

        # scale the workload uniformly to meet the min_processed_fraction requirement
        contribs .= simulate(ps) .* θs ./ ps
        s = sum(contribs)
        ps ./= min_processed_fraction / s
        contribs .*= min_processed_fraction / s

        # estimate gradients with respect to each element
        for i in 1:nworkers
            ∇s[i] = finite_diff(ps, i)
        end

        # make one pass over the workers, updating the number of partitions
        for i in 1:nworkers
            if isapprox(contribs[i], 0) || isapprox(∇s[i], 0)
                # the contribution or gradient is zero when the worker never participates, 
                # so we need to increase the number of partitions                            
                x = (1+α)*ps[i]
            else
                # solve for x in contribs[i] + (x - ps[i]) * ∇s[i] - μ = 0
                # (Newton's method)
                x = (μ - contribs[i]) / ∇s[i] + ps[i]
                x = min(x, (1+α)*ps[i])
                x = max(x, (1-α)*ps[i])
                x = max(x, 1.0)
            end
            ps[i] = x
        end
        t = time_ns()
    end

    # scale the workload uniformly to meet the min_processed_fraction requirement
    contribs .= simulate(ps) .* θs ./ ps
    s = sum(contribs)
    ps ./= min_processed_fraction / s
    contribs .*= min_processed_fraction / s
    loss = var(contribs)

    ps, loss
end

function optimize(sim::EventDrivenSimulator, qs0; θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_processed_fraction::Real, populationSize::Integer=100, tournamentSize::Integer=32, mutationRate::Real=1.0, time_limit::Real=10.0, simulation_niterations::Integer=100, simulation_nsamples::Integer=10)
    0 < min_processed_fraction <= 1 || throw(ArgumentError("min_processed_fraction is $min_processed_fraction"))
    nworkers = length(qs0)
    length(θs) == nworkers || throw(DimensionMismatch("θs has dimension $(length(θs)), but nworkers is $nworkers"))
    length(comp_mcs) == nworkers || throw(DimensionMismatch("comp_mcs has dimension $(length(comp_mcs)), but nworkers is $nworkers"))
    length(comp_vcs) == nworkers || throw(DimensionMismatch("comp_vcs has dimension $(length(comp_vcs)), but nworkers is $nworkers"))
    length(comm_mcs) == nworkers || throw(DimensionMismatch("comm_mcs has dimension $(length(comm_mcs)), but nworkers is $nworkers"))
    length(comm_vcs) == nworkers || throw(DimensionMismatch("comm_vcs has dimension $(length(comm_vcs)), but nworkers is $nworkers"))    

    # setup communication latency distributions
    for i in 1:nworkers
        m = comm_mcs[i]
        v = comm_vcs[i]
        sim.comm_distributions[i] = distribution_from_mean_variance(Gamma, m, v)
    end

    # simulation helper function
    ls = zeros(nworkers)
    function simulate(qs)

        # @info "evaluating $(1 ./ qs)"

        # setup compute latency distributions
        for i in 1:nworkers
            m = comp_mcs[i] * θs[i] * qs[i]
            v = comp_vcs[i] * θs[i] * qs[i]
            sim.comp_distributions[i] = distribution_from_mean_variance(Gamma, m, v)
        end
    
        # run nsamples simulations, each consisting of niterations steps
        ls .= 0
        for _ in 1:simulation_nsamples
            sim = EventDrivenSimulator(sim)
            step!(sim, simulation_niterations)
            ls .+= sim.nfresh ./ simulation_niterations
        end
        ls ./= simulation_nsamples
    end

    # constraint
    # (the total expected conbtribution must be above some threshold)    
    function g(qs, ls)
        rv = 0.0
        for i in 1:nworkers
            rv += ls[i] * θs[i] * qs[i]
        end
        rv - min_processed_fraction
    end

    # objective function
    # (the variance of the contribution between workers)
    lk = ReentrantLock() # objective function isn't thread-safe
    fworst = 0.0 # worst objective function value observed so far
    function f(qs)
        rv = 0.0
        lock(lk) do
            ls = simulate(qs)
            c = g(qs, ls)
            if c < 0
                rv = fworst + abs(c)
            else
                ls .*= θs .* qs
                rv = var(ls)
                fworst = max(fworst, rv)
            end
        end
        rv
    end    

    # evolutionary algorithm setup
    selection = Evolutionary.tournament(tournamentSize)
    crossover = Evolutionary.LX()
    lower = max.(0.0, qs0 .* 0.5)
    upper = min.(1.0, qs0 .* 2.0)
    mutation = Evolutionary.domainrange((lower .- upper) ./ 10) # as recommended in the BGA paper

    # wraps a mutation, but ensures that the inverse of each element is integer
    function integer_mutation(qs)
        if isone(mutationRate) || rand() < mutationRate
            mutation(qs)
        end
        for i in 1:length(qs)
            p = 1 / qs[i]            
            if rand() < 0.5
                p = floor(p)
            else
                p = ceil(p)
            end
            p = max(p, 1.0)
            qs[i] = 1 / p
        end
        qs
    end

    # for reference, compute objective function value for the initial solution
    f0 = f(qs0)
    f0 = iszero(fworst) ? NaN : f0

    # optimization algorithm
    opt = Evolutionary.GA(;populationSize, mutationRate=1.0, selection, crossover, mutation=integer_mutation)
    options = Evolutionary.Options(;time_limit, Evolutionary.default_options(opt)...)
    Evolutionary.optimize(f, lower, upper, qs0, opt, options), f0
end

function load_balancer(chin::Channel, chout::Channel; min_processed_fraction::Real, nwait::Integer, nworkers::Integer, time_limit::Real=10.0)
    0 < min_processed_fraction <= 1 || throw(ArgumentError("min_processed_fraction is $min_processed_fraction"))
    0 < nworkers || throw(ArgumentError("nworkers is $nworkers"))
    0 < nwait <= nworkers || throw(ArgumentError("nwait is $nwait, but nworkers is $nworkers"))
    @info "load_balancer task started"

    # fraction of dataset stored by and fraction of local data processed per iteration for each worker
    θs = fill(NaN, nworkers)
    qs = fill(NaN, nworkers)
    ps = fill(NaN, nworkers)

    # mean and variance coefficients for each worker
    comp_mcs = fill(NaN, nworkers)
    comp_vcs = fill(NaN, nworkers)
    comm_mcs = fill(NaN, nworkers)
    comm_vcs = fill(NaN, nworkers)

    # reusable simulator
    comp_distributions = Vector{Gamma}(undef, nworkers)
    comm_distributions = Vector{Gamma}(undef, nworkers)
    sim = EventDrivenSimulator(;nwait, nworkers, comp_distributions, comm_distributions)

    # process an output received from the profiler
    function process_sample(v::ProfilerOutput)
        0 < v.worker <= nworkers || throw(ArgumentError("v.worker is $(v.worker), but nworkers is $nworkers"))
        0 <= v.θ <= 1 || throw(ArgumentError("θ is $θ"))
        0 <= v.q <= 1 || throw(ArgumentError("q is $q"))
        0 <= v.comp_mc || throw(ArgumentError("comp_mc is $comp_mc"))        
        0 <= v.comp_vc || throw(ArgumentError("comp_vc is $comp_vc"))        
        0 <= v.comm_mc || throw(ArgumentError("comm_mc is $comm_mc"))
        0 <= v.comm_vc || throw(ArgumentError("comm_vc is $comm_vc"))
        isnan(v.θ) || (θs[v.worker] = v.θ)
        isnan(v.q) || (qs[v.worker] = v.q)
        isnan(v.comp_mc) || (comp_mcs[v.worker] = v.comp_mc)
        isnan(v.comp_vc) || (comp_vcs[v.worker] = v.comp_vc)
        isnan(v.comm_mc) || (comm_mcs[v.worker] = v.comm_mc)
        isnan(v.comm_vc) || (comm_vcs[v.worker] = v.comm_vc)        
        return
    end

    # helper to check if there is any missing latency data
    # (in which case we shouldn't run the load-balancer)
    all_populated = false    
    function check_populated()
        if all_populated
            return all_populated
        end        
        all_populated = iszero(count(isnan, comp_mcs))
        if all_populated
            all_populated = all_populated && iszero(count(isnan, comp_vcs))
        end
        if all_populated
            all_populated = all_populated && iszero(count(isnan, comm_mcs))
        end
        if all_populated
            all_populated = all_populated && iszero(count(isnan, comm_vcs))
        end
        if all_populated
            all_populated = all_populated && iszero(count(isnan, θs))
        end
        if all_populated
            all_populated = all_populated && iszero(count(isnan, qs))
        end
        all_populated
    end

    # consume incoming ProfilerOutput samples and run the optimizer
    while isopen(chin)

        # consume all values currently in the channel
        try
            vin = take!(chin)
            process_sample(vin)
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
                process_sample(vin)
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
        if !check_populated()
            continue
        end

        # initialization
        if count(isnan, ps) > 0
            ps .= 1 ./ qs
        end

        # new = balance_contribution(ps, min_processed_fraction; θs, ds_comm, cms_comp, cvs_comp, nwait)
        try
            # @info "running load-balancer w. ps: $ps, θs: $θs, comp_mcs: $comp_mcs, comp_vcs: $comp_vcs"
            t = @elapsed begin
                result, f0 = optimize(sim, 1.0./ps; θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_processed_fraction, time_limit)
            end
            new_qs = result.minimizer
            @info "load-balancer finished in $t seconds"

            # compare the initial and new solutions, and continue if the change isn't large enough
            if !isnan(f0) && minimum(result) > f0 * 0.9
                @info "load-balancer result of $(minimum(result)) not sufficiently better than $f0; continuing"
                continue
            end

            # push any changes into the output channel
            for i in 1:nworkers
                p = round(Int, 1/new_qs[i])
                if p != ps[i]
                    ps[i] = p
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