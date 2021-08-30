using StatsBase, Statistics, Dates

# setup
nworkers = 2
windowsize = Second(10)
qlower, qupper = 0.0, 1.0
chin, chout = CodedComputing.setup_profiler_channels()

function f(vs)
    l, u = quantile(vs, qlower), quantile(vs, qupper)
    filter((x)->l<=x<=u, vs)
end

# put example values into the input channel
## worker 1
timestamps1 = Time(0) .+ [Second(i) for i in 1:11]
comps1 = rand(length(timestamps1))
comms1 = rand(length(timestamps1))
for i in 1:length(timestamps1)
    v = @NamedTuple{worker::Int,timestamp::Time,comp::Float64,comm::Float64}((1, timestamps1[i], comps1[i], comms1[i]))
    push!(chin, v)
end
comps1 = comps1[end-1:end]
comms1 = comms1[end-1:end]
comp_mean, comp_var = mean(comps1), var(comps1, corrected=true)
comm_mean, comm_var = mean(comms1), var(comms1, corrected=true)
correct1 = @NamedTuple{worker::Int,comp_mean::Float64,comp_var::Float64,comm_mean::Float64,comm_var::Float64}((1,comp_mean,comp_var,comm_mean,comm_var))

## worker 2
timestamps2 = Time(0) .+ [Second(i) for i in 1:20]
comps2 = rand(length(timestamps2))
comms2 = rand(length(timestamps2))
for i in 1:length(timestamps2)
    v = @NamedTuple{worker::Int,timestamp::Time,comp::Float64,comm::Float64}((2, timestamps2[i], comps2[i], comms2[i]))
    push!(chin, v)
end
comps2 = f(comps2[end-10:end])
comms2 = f(comms2[end-10:end])
comp_mean, comp_var = mean(comps2), var(comps2, corrected=true)
comm_mean, comm_var = mean(comms2), var(comms2, corrected=true)
correct2 = @NamedTuple{worker::Int,comp_mean::Float64,comp_var::Float64,comm_mean::Float64,comm_var::Float64}((2,comp_mean,comp_var,comm_mean,comm_var))

# start the task and test that it consumes the input values
task = Threads.@spawn CodedComputing.latency_profiler(chin, chout; nworkers, qlower, qupper, windowsize)

# sleep(10)

# wait for up to 10 seconds for the input to be consumed
t0 = time_ns()
while (time_ns() - t0)/1e9 < 10 && isready(chin)
    sleep(0.1)
end
@test !isready(chin)

# wait for up to 10 seconds for the subsystem to produce output
t0 = time_ns()
while (time_ns() - t0)/1e9 < 10 && !isready(chout)
    sleep(0.1)
end

# there should be 2 values in the output channel
## test the first output value
@test isready(chout)
v = take!(chout)
@test v.worker == 1 || v.worker == 2
if v.worker == 1
    @test all(isapprox.(values(v), values(correct1)))
elseif v.worker == 2
    @test all(isapprox.(values(v), values(correct2)))
end

## test the second output value
@test isready(chout)
v = take!(chout)
@test v.worker == 1 || v.worker == 2
if v.worker == 1
    @test all(isapprox.(values(v), values(correct1)))
elseif v.worker == 2
    @test all(isapprox.(values(v), values(correct2)))
end

close(chin)
close(chout)