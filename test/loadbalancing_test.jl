using Distributions

@testset "smoke-test" begin
    chin, chout = CodedComputing.setup_loadbalancer_channels()

    nworkers = 2
    nwait = 1
    min_processed_fraction = 0.1
    time_limit = 1.0 # must be floating-point
    θs = [0.3, 0.7]
    qs = 1 ./ [2, 3]

    # put some random values into the load-balancer input
    Random.seed!(123)
    worker = 1
    v1 = CodedComputing.ProfilerOutput(worker, θs[worker], qs[worker], rand(), rand(), rand(), rand())
    push!(chin, v1)

    worker = 2
    v2 = CodedComputing.ProfilerOutput(worker, θs[worker], qs[worker], rand(), rand(), rand(), rand())
    push!(chin, v2)

    # start the load-balancer
    task = Threads.@spawn CodedComputing.load_balancer(chin, chout; min_processed_fraction, nwait, nworkers, time_limit)

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

@testset "optimizer" begin
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

    ps, loss = CodedComputing.optimize!(ps, sim; θs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_processed_fraction, time_limit=5)
    slows = ps[1:nslow]
    slows_mean = mean(slows)
    fasts = ps[nslow+1:end]
    fasts_mean = mean(fasts)
    @test 2 / mean(slows) ≈ 1 / mean(fasts) rtol=0.1
    for v in slows
        @test v ≈ slows_mean rtol=0.1
    end
    for v in fasts
        @test v ≈ fasts_mean rtol=0.1
    end
end