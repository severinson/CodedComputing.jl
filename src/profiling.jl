# Code used for latency profiling
using StatsBase, Statistics, OnlineStats, Dates

function setup_profiler_channels(;chin_size=Inf, chout_size=Inf)
    chin = Channel{@NamedTuple{worker::Int,timestamp::Time,comp::Float64,comm::Float64}}(chin_size)
    chout = Channel{@NamedTuple{worker::Int,comp_mean::Float64,comp_var::Float64,comm_mean::Float64,comm_var::Float64}}(chout_size)
    chin, chout
end

"""

Latency profiling sub-system. Receives latency observations on `chin`, computes the mean and 
variance over a moving time window of length `windowsize`, and sends the results on `chout`.
"""
function latency_profiler(
    chin::Channel{@NamedTuple{worker::Int,timestamp::Time,comp::Float64,comm::Float64}}, 
    chout::Channel{@NamedTuple{worker::Int,comp_mean::Float64,comp_var::Float64,comm_mean::Float64,comm_var::Float64}};
    nworkers::Integer, qlower::Real=0.1, qupper::Real=0.9, buffersize::Integer=1000, windowsize::Dates.AbstractTime=Second(60))
    @info "latency_profiler task started"    
    
    # maintain a window of latency samples for each worker
    ws = [MovingTimeWindow(windowsize, valtype=@NamedTuple{comp::Float64, comm::Float64}, timetype=Time) for _ in 1:nworkers]

    # utility function for updating the correct window with a new latency measurement
    function fitwindow(v)
        0 < v.worker <= nworkers || @error "Expected v.worker to be in [1, $nworkers], but it is $(v.worker)"
        if v.comp < 0 || v.comm < 0
            @error "latency_profiler received negative latency sample" v
            return
        end
        isnan(v.comm) || isnan(v.comp) || fit!(ws[v.worker], (v.timestamp, @NamedTuple{comp::Float64,comm::Float64}((v.comp, v.comm))))
        return
    end

    # helper functions for computing the mean and variance over the samples in a window
    buffer = zeros(buffersize)
    function processwindow(w::MovingTimeWindow, key::Symbol)
        key == :comp || key == :comm || throw(ArgumentError("key must be either :comp or :comm, but is $key"))        

        # populate the buffer
        i = 0
        n = 0
        for (_, t) in value(w)
            v = t[key]
            if !isnan(v)
                buffer[i+1] = v
                i = mod(i + 1, buffersize)
                n += 1
            end
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
        if length(vs) == 0
            return NaN, NaN
        end
        m = mean(vs)
        v = var(vs, mean=m, corrected=true)        
        m, v
    end

    # process incoming latency samples
    while isopen(chin)

        # consume all values currently in the channel
        try
            vin = take!(chin)
            fitwindow(vin)
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
                fitwindow(vin)
            catch e
                if e isa InvalidStateException
                    @info "error taking value from input channel" e
                    break
                else
                    rethrow()
                end
            end
        end        

        for i in 1:nworkers

            # for each worker, compute the mean and variance of the values in the window            
            comp_mean, comp_var = processwindow(ws[i], :comp)
            comm_mean, comm_var = processwindow(ws[i], :comm)
            if isnan(comp_mean) || isnan(comp_var) || isnan(comm_mean) || isnan(comm_var)
                continue
            end

            # push the computed statistics into the output channel
            vout = @NamedTuple{worker::Int,comp_mean::Float64,comp_var::Float64,comm_mean::Float64,comm_var::Float64}((i,comp_mean,comp_var,comm_mean,comm_var))            
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