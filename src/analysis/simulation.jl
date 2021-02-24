
"""

Simulate compute latency of a single worker
"""
function simulate_worker_latency!(samples::AbstractVector; worker_flops=7.56e7)
    @assert isapprox(worker_flops, 7.56e7, rtol=1e-2)
    
    # mean latency for this worker
    μ = 0.15185292155456753
    σ = 0.004095785673988278
    Distributions.rand!(Normal(μ, σ), samples)

    # latency distribution outside of bursts
    μ = rand(Normal(2.68559984516683e-18, 4.272504064627912e-16))
    σ = rand(LogNormal(-14.11655969017424, 0.6843218110446242))
    noise_rv = Normal(μ, σ)

    # latency distribution during bursts
    μ = rand(Normal(0.0221600607045976, 0.0014171964533128108))
    σ = rand(LogNormal(-9.495472609830145, 0.19837228407780638))
    burst_rv = Normal(μ, σ)

    # initial state (1 corresponds to outside of a burst and 2 to during a burst)
    state = rand() < 0.985165 ? 1 : 2

    # latency series
    for i in 1:length(samples)
        if state == 1 # outside of a burst
            samples[i] += rand(noise_rv)
            if rand() < 7.00203e-5
                state = 2
            end
        else # during a burst
            samples[i] += rand(burst_rv)
            if rand() < 0.00464986
                state = 1
            end
        end
    end
    samples
end

"""

Simulate latency of several worker, with each column of `samples` corresponding to a worker
"""
function simulate_worker_latency!(samples::AbstractMatrix)
    for i in 1:size(samples, 2)
        simulate_worker_latency!(view(samples, :, i))
    end
    samples
end

"""

Simulate order statistics latency for `nworkers` workers when waiting for the fastest `nwait` workers in each iteration
"""
function simulate_orderstats(niterations::Integer, nsamples::Integer, nworkers::Integer, nwait::Integer)
    samples = zeros(niterations, nworkers)
    # rv = zeros(niterations)
    rv = 0.0
    for _ in 1:nsamples
        simulate_worker_latency!(samples)
        sort!(samples, dims=2)
        rv += mean(view(samples, :, nwait))
    end
    rv /= nsamples
end