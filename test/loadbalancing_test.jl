using Distributions

@testset "less_than_lower_bound" begin
    nworkers = 9
    nsubpartitions = 160.0
    θs = fill(1/nworkers, nworkers)
    ps = fill(nsubpartitions, nworkers)

    # values recorded on eX3 with low variance and very low comm. latency relative to comp.
    comp_mcs = [255.07814185599995, 235.33940168554452, 253.72451957702958, 116.82766857029705, 116.75048824871284, 116.63288112475249, 126.99890071128709, 126.4382596942574, 117.12455760475244]
    comp_vcs = [0.0027557071631392205, 0.003592251480945663, 0.002281988669487589, 0.0007248118462346298, 0.0005408877863892769, 0.0005133790987450482, 0.0005865578758950107, 0.0006378358097673953, 0.0005586345559702721]    
    comm_mcs = [3.373097029702779e-5, 3.189267326732839e-5, 3.427457425742762e-5, 3.50826633663372e-5, 3.664775247524789e-5, 3.760737623762412e-5, 3.5680287128712425e-5, 3.737996039603971e-5, 3.6443900990098125e-5]
    comm_vcs = [1.4755830783262175e-12, 1.5582646160146614e-12, 1.3494372741908877e-12, 2.0233815302431064e-12, 2.7092806020993726e-12, 2.554836195077552e-12, 1.7668490363708934e-12, 1.5867281370452314e-12, 1.8507092575245513e-12]    
   
    dzs = CodedComputing.distribution_from_mean_variance.(Gamma, comp_mcs .* θs ./ ps, comp_vcs .* θs ./ ps)
    dys = CodedComputing.distribution_from_mean_variance.(Gamma, comm_mcs, comm_vcs)

    correct = [-6787.780734619605, -5997.6993965555075, -7030.083983125667, -4542.376367667996, -4266.766708613465, -4204.885092182273, -4709.103013914913, -4746.332511515265, -4315.852500222745]
    @test CodedComputing.less_than_lower_bound(dzs, dys) ≈ correct
end

@testset "optimizer (comp. higher than comm.)" begin
    nworkers = 10
    nwait = nworkers
    nslow = 3    
    nsubpartitions = 160.0
    θs = fill(1/nworkers, nworkers)
    ps = fill(nsubpartitions, nworkers)

    # per-worker compute latency
    comp_ms = fill(1.0, nworkers)
    comp_ms[1:nslow] .*= 2    
    comp_vs = comp_ms ./ 100
    comp_mcs = comp_ms ./ (θs ./ ps)
    comp_vcs = comp_vs ./ (θs ./ ps)

    # per-worker communication latency
    comm_mcs = fill(1e-2, nworkers)
    comm_vcs = comm_mcs ./ 100
    
    sim_nwait = floor(Int, nworkers/2)
    comp_distributions = [Gamma() for _ in 1:nworkers]
    comm_distributions = [Gamma() for _ in 1:nworkers]
    sim = EventDrivenSimulator(;nwait=sim_nwait, nworkers, comm_distributions, comp_distributions)
    min_processed_fraction = sim_nwait / nworkers / nsubpartitions

    ps, loss, loss0 = CodedComputing.optimize!(ps, ps, sim; θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_processed_fraction, time_limit=2)
    @test loss < Inf    

    # check that the slow workers have about twice as many partitions
    slows = ps[1:nslow]
    slows_mean = mean(slows)
    fasts = ps[nslow+1:end]
    fasts_mean = mean(fasts)
    @test 2 / mean(slows) ≈ 1 / mean(fasts) rtol=0.1

    # check that all slow and all fast workers were given approx. the same number of partitions as each other
    for v in slows
        @test v ≈ slows_mean rtol=0.1
    end
    for v in fasts
        @test v ≈ fasts_mean rtol=0.1
    end

    # check that the avg. latency is uniform (within some margin)
    ms = comp_mcs ./ ps .+ comm_mcs
    μ = mean(ms)
    for i in 1:nworkers
        @test ms[i] ≈ μ rtol=0.1
    end
end

@testset "optimizer (comp. much higher than comm.)" begin
    nworkers = 9
    nwait = nworkers
    nsubpartitions = 160.0
    θs = fill(1/nworkers, nworkers)
    ps = fill(nsubpartitions, nworkers)

    # values recorded on eX3 with low variance and very low comm. latency relative to comp.
    comp_mcs = [255.07814185599995, 235.33940168554452, 253.72451957702958, 116.82766857029705, 116.75048824871284, 116.63288112475249, 126.99890071128709, 126.4382596942574, 117.12455760475244]
    comp_vcs = [0.0027557071631392205, 0.003592251480945663, 0.002281988669487589, 0.0007248118462346298, 0.0005408877863892769, 0.0005133790987450482, 0.0005865578758950107, 0.0006378358097673953, 0.0005586345559702721]    
    comm_mcs = [3.373097029702779e-5, 3.189267326732839e-5, 3.427457425742762e-5, 3.50826633663372e-5, 3.664775247524789e-5, 3.760737623762412e-5, 3.5680287128712425e-5, 3.737996039603971e-5, 3.6443900990098125e-5]
    comm_vcs = [1.4755830783262175e-12, 1.5582646160146614e-12, 1.3494372741908877e-12, 2.0233815302431064e-12, 2.7092806020993726e-12, 2.554836195077552e-12, 1.7668490363708934e-12, 1.5867281370452314e-12, 1.8507092575245513e-12]    
    
    sim_nwait = floor(Int, nworkers/2)
    comp_distributions = [Gamma() for _ in 1:nworkers]
    comm_distributions = [Gamma() for _ in 1:nworkers]    
    sim = EventDrivenSimulator(;nwait=sim_nwait, nworkers, comm_distributions, comp_distributions)
    min_processed_fraction = sim_nwait / nworkers / nsubpartitions

    ps, loss, loss0 = CodedComputing.optimize!(ps, ps, sim; θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_processed_fraction, time_limit=2)
    @test loss < Inf

    # check that the avg. latency is uniform (within some margin)    
    ms = comp_mcs ./ ps .+ comm_mcs
    μ = mean(ms)
    for i in 1:nworkers
        @test ms[i] ≈ μ rtol=0.1
    end
end

@testset "optimizer (comp. same as comm.)" begin
    nworkers = 9
    nwait = nworkers
    nsubpartitions = 160.0
    θs = fill(1/nworkers, nworkers)
    ps = fill(nsubpartitions, nworkers)

    # values recorded on eX3 with low variance and very low comm. latency relative to comp.
    comp_mcs = [255.07814185599995, 235.33940168554452, 253.72451957702958, 116.82766857029705, 116.75048824871284, 116.63288112475249, 126.99890071128709, 126.4382596942574, 117.12455760475244]
    comp_vcs = [0.0027557071631392205, 0.003592251480945663, 0.002281988669487589, 0.0007248118462346298, 0.0005408877863892769, 0.0005133790987450482, 0.0005865578758950107, 0.0006378358097673953, 0.0005586345559702721]    
    comm_mcs = copy(comp_mcs)
    comm_vcs = copy(comp_vcs)
    
    sim_nwait = floor(Int, nworkers/2)
    comp_distributions = [Gamma() for _ in 1:nworkers]
    comm_distributions = [Gamma() for _ in 1:nworkers]        
    sim = EventDrivenSimulator(;nwait=sim_nwait, nworkers, comm_distributions, comp_distributions)
    min_processed_fraction = sim_nwait / nworkers / nsubpartitions

    ps, loss, loss0 = CodedComputing.optimize!(ps, ps, sim; θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_processed_fraction, time_limit=2)
    @test loss < Inf

    # check that the avg. latency is uniform (within some margin)    
    ms = comp_mcs ./ ps .+ comm_mcs
    μ = mean(ms)
    for i in 1:nworkers
        @test ms[i] ≈ μ rtol=0.5
    end
end

@testset "smoke-test" begin
    chin, chout = CodedComputing.setup_loadbalancer_channels()

    nworkers = 2
    nwait = 1
    min_processed_fraction = 0.1
    time_limit = 1.0 # must be floating-point
    θs = [0.3, 0.7]
    qs = 1 ./ [2, 3]
    ps = round.(Int, 1 ./ qs)

    # put some random values into the load-balancer input
    Random.seed!(123)
    worker = 1
    v1 = CodedComputing.ProfilerOutput(worker, θs[worker], qs[worker], rand(), rand(), rand(), rand())
    push!(chin, v1)

    worker = 2
    v2 = CodedComputing.ProfilerOutput(worker, θs[worker], qs[worker], rand(), rand(), rand(), rand())
    push!(chin, v2)

    # start the load-balancer
    task = Threads.@spawn CodedComputing.load_balancer(chin, chout; min_processed_fraction, nwait, nsubpartitions=ps, nworkers, time_limit, min_improvement=1)

    # wait for up to 10 seconds for the input to be consumed
    t0 = time_ns()
    while (time_ns() - t0)/1e9 < 10 && isready(chin)
        sleep(0.1)
    end
    if istaskfailed(task)
        wait(task)
    end
    @test !isready(chin)

    # wait for up to 10 seconds for the subsystem to produce output
    t0 = time_ns()
    while (time_ns() - t0)/1e9 < 10 && !isready(chout)
        sleep(0.1)
    end
    if istaskfailed(task)
        wait(task)
    end

    correct1 = (1, 1)
    correct2 = (2, 7)

    @test isready(chout)
    vout = take!(chout)
    correct = vout.worker == 1 ? correct1 : correct2
    # @test vout == correct
    println(vout)

    @test isready(chout)
    vout = take!(chout)
    correct = vout.worker == 1 ? correct1 : correct2
    # @test vout == correct
    println(vout)

    # stop the profiler
    close(chin)
    close(chout)
    wait(task)
end