# Code used for latency profiling
using StatsBase, Statistics, Dates

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

function StatsBase.var(f::Function, itr)
    g = (x) -> f(x)^2
    mean(g, itr) - mean(f, itr)^2
end

"""

Remove all values at the end of the window older than windowsize, and return the number of 
elements removed.
"""
function Base.filter!(w::CircularBuffer{ProfilerInput}; windowsize)
    rv = 0
    while length(w) > 0 && (w[1].timestamp - w[end].timestamp) > windowsize
        pop!(w)
        rv += 1
    end
    rv
end

"""

Return a view into the elements of w in beween the qlower and qupper quantiles.
"""
function comp_quantile_view(w::CircularBuffer{ProfilerInput}, buffer::Vector{ProfilerInput}, qlower::Real, qupper::Real)
    0 <= qlower <= qupper <= 1.0 || throw(ArgumentError("qlower is $qlower and qupper is $qupper"))
    n = length(w)
    for i in 1:n
        buffer[i] = w[i]
    end
    @views sort!(buffer[1:n], by=(x)->getfield(x, :comp_delay), alg=QuickSort)
    il = max(1, ceil(Int, n*qlower))
    iu = min(length(buffer), floor(Int, n*qupper))
    view(buffer, il:iu)
end

"""

Return a view into the elements of w in beween the qlower and qupper quantiles.
"""
function comm_quantile_view(w::CircularBuffer{ProfilerInput}, buffer::Vector{ProfilerInput}, qlower::Real, qupper::Real)
    0 <= qlower <= qupper <= 1.0 || throw(ArgumentError("qlower is $qlower and qupper is $qupper"))
    n = length(w)
    for i in 1:n
        buffer[i] = w[i]
    end
    @views sort!(buffer[1:n], by=(x)->getfield(x, :comm_delay), alg=QuickSort)
    il = max(1, ceil(Int, n*qlower))
    iu = min(length(buffer), floor(Int, n*qupper))
    view(buffer, il:iu)
end

function comp_mean_var(w::CircularBuffer{ProfilerInput}; buffer::Vector{ProfilerInput}, qlower::Real, qupper::Real, minsamples::Integer)
    vs = comp_quantile_view(w, buffer, qlower, qupper)
    if length(vs) < minsamples
        return NaN, NaN
    end    
    m = mean((x)->getfield(x, :comp_delay) / (getfield(x, :θ) * getfield(x, :q)), vs)
    v = var((x)->getfield(x, :comp_delay) / sqrt(getfield(x, :θ) * getfield(x, :q)), vs)    
    m, v
end

function comm_mean_var(w::CircularBuffer{ProfilerInput}; buffer::Vector{ProfilerInput}, qlower::Real, qupper::Real, minsamples::Integer)
    vs = comm_quantile_view(w, buffer, qlower, qupper)
    if length(vs) < minsamples
        return NaN, NaN
    end
    m = mean((x)->getfield(x, :comm_delay), vs)
    v = var((x)->getfield(x, :comm_delay), vs)
    m, v
end

function process_window(w::CircularBuffer{ProfilerInput}, i::Integer; buffer::Vector{ProfilerInput}, qlower::Real, qupper::Real, minsamples::Integer)::ProfilerOutput
    length(w) > 0 || throw(ArgumentError("window must not be empty"))
    θ = mean((x)->getfield(x, :θ), w)
    q = mean((x)->getfield(x, :q), w)
    comp_mc, comp_vc = comp_mean_var(w; buffer, qlower, qupper, minsamples)
    comm_mc, comm_vc = comm_mean_var(w; buffer, qlower, qupper, minsamples)
    ProfilerOutput(i, θ, q, comp_mc, comp_vc, comm_mc, comm_vc)
end

"""

Latency profiling sub-system. Receives latency observations on `chin`, computes the mean and 
variance over a moving time window of length `windowsize`, and sends the results on `chout`.
"""
function latency_profiler(chin::Channel{ProfilerInput}, chout::Channel{ProfilerOutput}; nworkers::Integer, qlower::Real=0.1, qupper::Real=0.9, buffersize::Integer=1000, minsamples::Integer=10, windowsize::Dates.AbstractTime=Second(60))
    0 < nworkers || throw(ArgumentError("nworkers is $nworkers"))
    0 <= qlower <= qupper <= 1.0 || throw(ArgumentError("qlower is $qlower and qupper is $qupper"))
    @info "latency_profiler task started on thread $(Threads.threadid())"
    
    # maintain a window of latency samples for each worker
    ws = [CircularBuffer{CodedComputing.ProfilerInput}(buffersize) for _ in 1:nworkers]
    buffer = Vector{ProfilerInput}(undef, buffersize)

    # process incoming latency samples
    while isopen(chin)

        # consume all values currently in the channel
        try
            vin::ProfilerInput = take!(chin)
            if !isnan(vin.comp_delay) && !isnan(vin.comm_delay)
                pushfirst!(ws[vin.worker], vin)
            end
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
                vin::ProfilerInput = take!(chin)
                if !isnan(vin.comp_delay) && !isnan(vin.comm_delay)
                    pushfirst!(ws[vin.worker], vin)
                end
            catch e
                if e isa InvalidStateException
                    @info "error taking value from input channel" e
                    break
                else
                    rethrow()
                end
            end
        end

        # filter out values older than windowsize
        for i in 1:nworkers
            filter!(ws[i]; windowsize)
        end
        
        # remove any values already in the output channel before putting new ones in
        while isready(chout)
            take!(chout)
        end

        # compute updated statistics for all workers
        for i in 1:nworkers
            if length(ws[i]) == 0
                continue
            end
            vout = process_window(ws[i], i; buffer, qlower, qupper, minsamples)
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