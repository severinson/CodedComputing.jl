# Code used for latency profiling
using StatsBase, Statistics, OnlineStats, Dates

struct ProfilerInput
    worker::Int
    θ::Float64
    q::Float64
    timestamp::Time    
    comp_delay::Float64
    comm_delay::Float64    
end

struct ProfilerOutput
    worker::Int         # worker index
    θ::Float64          # fraction of the dataset stored by this worker, averaged over all input samples that make up this output
    q::Float64          # fraction of local data processed per iteration, averaged over all input samples that make up this output
    comp_mc::Float64    # = (mean comp. delay) / (θ*q)
    comp_vc::Float64    # = (var of comp. delay) / (θ*q)
    comm_mc::Float64    # = mean comm. delay
    comm_vc::Float64    # = var of comm. delay
end

Base.isless(p::Pair{Time, CodedComputing.ProfilerInput}, q::Pair{Time, CodedComputing.ProfilerInput}) = isless(first(p), first(q))

function setup_profiler_channels(;chin_size=Inf, chout_size=Inf)
    chin = Channel{ProfilerInput}(chin_size)
    chout = Channel{ProfilerOutput}(chout_size)
    chin, chout
end

"""

Latency profiling sub-system. Receives latency observations on `chin`, computes the mean and 
variance over a moving time window of length `windowsize`, and sends the results on `chout`.
"""
function latency_profiler(chin::Channel{ProfilerInput}, chout::Channel{ProfilerOutput}; nworkers::Integer, qlower::Real=0.1, qupper::Real=0.9, buffersize::Integer=1000, minsamples::Integer=10, windowsize::Dates.AbstractTime=Second(60))
    0 < nworkers || throw(ArgumentError("nworkers is $nworkers"))
    0 <= qlower <= qupper <= 1.0 || throw(ArgumentError("qlower is $qlower and qupper is $qupper"))
    @info "latency_profiler task started"
    
    # maintain a window of latency samples for each worker
    ws = [MovingTimeWindow(windowsize, valtype=ProfilerInput, timetype=Time) for _ in 1:nworkers]

    # utility function for updating the correct window with a new latency measurement
    function process_sample(v)
        0 < v.worker <= nworkers || throw(ArgumentError("v.worker is $(v.worker), but nworkers is $nworkers"))
        0 <= v.θ <= 1 || throw(ArgumentError("θ is $θ"))
        0 <= v.q <= 1 || throw(ArgumentError("q is $q"))
        0 <= v.comp_delay || throw(ArgumentError("comp_delay is $comp_delay"))
        0 <= v.comm_delay || throw(ArgumentError("comm_delay is $comm_delay"))
        if isnan(v.comp_delay) || isnan(v.comm_delay)
            return
        end
        fit!(ws[v.worker], (v.timestamp, v))
        return
    end

    # compute the mean over all values in the window
    # function window_mean(w::MovingTimeWindow, key::Symbol)¨
    function window_mean(w, key)
        rv = 0.0
        n = 0
        for (_, t) in OnlineStats.value(w)
            rv += getfield(t, key)
            n += 1
        end
        rv / n
    end

    # compute the mean and variance over all values in the window between the qlower and qupper quantiles
    buffer = zeros(buffersize)
    function window_mean_var(w::MovingTimeWindow, key::Symbol)

        # populate the buffer
        i = 0
        n = 0
        for (_, t) in OnlineStats.value(w)
            v = getfield(t, key)

            # compute delay should be normalized
            if key == :comp_delay
                v /= (getfield(t, :θ) * getfield(t, :q))
            end            

            if isnan(v)
                continue
            end
            buffer[i+1] = v
            i = mod(i + 1, buffersize)
            n += 1
        end
        n = min(n, buffersize)

        # return NaNs if there are no values
        if n == 0
            return NaN, NaN
        end

        # compute quantile indices
        sort!(view(buffer, 1:n))
        il = max(1, ceil(Int, n*qlower))
        iu = min(buffersize, floor(Int, n*qupper))

        # compute mean and variance over the values between qlower and qupper
        vs = view(buffer, il:iu)
        if length(vs) < minsamples
            return NaN, NaN
        end
        m = mean(vs)
        v = var(vs, mean=m, corrected=true)
        m, v
    end

    # process all samples in the window for the i-th worker
    function process_worker(i::Integer)
        0 < i <= nworkers || throw(ArgumentError("i is $i"))
        w = ws[i]
        θ = window_mean(w, :θ)
        q = window_mean(w, :q)
        comp_mc, comp_vc = window_mean_var(w, :comp_delay)
        comm_mc, comm_vc = window_mean_var(w, :comm_delay)
        ProfilerOutput(i, θ, q, comp_mc, comp_vc, comm_mc, comm_vc)
    end

    # process incoming latency samples
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

        # # to avoid overwhelming the consumer, only push new output if the channel is empty
        # if isready(chout)
        #     continue
        # end

        # compute updated statistics for all workers
        for i in 1:nworkers
            vout = process_worker(i)
            if isnan(vout.θ) || isnan(vout.q) || isnan(vout.comp_mc) || isnan(vout.comp_vc) || isnan(vout.comm_mc) || isnan(vout.comm_vc)
                @info "profiler dropped NaN-sample: $vout"
                continue
            end
            if vout.comp_mc < 0 || vout.comp_vc < 0 || vout.comm_mc < 0 || vout.comm_vc < 0
                @info "profiler dropped sample with negative latency: $vout"
                continue
            end
            if isapprox(vout.comp_mc, 0) || isapprox(vout.comp_vc, 0) || isapprox(vout.comm_mc, 0) || isapprox(vout.comm_vc, 0)
                @info "profiler dropped zero-sample: $vout"
                continue
            end
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
    @info "latency_profiler task finished"
end