using StatsBase, Statistics, Dates

# setup
nworkers = 2
windowsize = Second(10)
qlower, qupper = 0.0, 1.0
chin, chout = CodedComputing.setup_profiler_channels()
θs = [0.3, 0.7]
qs = 1 ./ [2, 3]

function f(vs)
    l, u = quantile(vs, qlower), quantile(vs, qupper)
    filter((x)->l<=x<=u, vs)
end

# put example values into the input channel
## worker 1
timestamps = Time(0) .+ [Second(i) for i in 1:11]
comps = rand(length(timestamps))
comms = rand(length(timestamps))
for i in 1:length(timestamps)
    v = CodedComputing.ProfilerInput(1, θs[1], qs[1], timestamps[i], comps[i], comms[i])
    push!(chin, v)
end
comps ./= θs[1] * qs[1]
comps = comps[end-1:end]
comms = comms[end-1:end]
comp_mean, comp_var = mean(comps), var(comps, corrected=true)
comm_mean, comm_var = mean(comms), var(comms, corrected=true)
correct1 = CodedComputing.ProfilerOutput(1, θs[1], qs[1], comp_mean, comp_var, comm_mean, comm_var)

## worker 2
timestamps = Time(0) .+ [Second(i) for i in 1:20]
comps = rand(length(timestamps))
comms = rand(length(timestamps))
for i in 1:length(timestamps)
    v = CodedComputing.ProfilerInput(2, θs[2], qs[2], timestamps[i], comps[i], comms[i])    
    push!(chin, v)
end
comps ./= θs[2] * qs[2]
comps = f(comps[end-10:end])
comms = f(comms[end-10:end])
comp_mean, comp_var = mean(comps), var(comps, corrected=true)
comm_mean, comm_var = mean(comms), var(comms, corrected=true)
correct2 = CodedComputing.ProfilerOutput(2, θs[2], qs[2], comp_mean, comp_var, comm_mean, comm_var)

# start the task and test that it consumes the input values
task = Threads.@spawn CodedComputing.latency_profiler(chin, chout; nworkers, qlower, qupper, windowsize)

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

# stop the profiler
close(chin)
wait(task)

# there should be 2 values in the output channel
## test the first output value
@test isready(chout)
v = take!(chout)
@test v.worker == 1 || v.worker == 2
correct = v.worker == 1 ? correct1 : correct2
for name in fieldnames(CodedComputing.ProfilerOutput)
    @test getfield(correct, name) ≈ getfield(v, name)
end
# @test all(isapprox.(values(v), values(correct1)))

## test the second output value
@test isready(chout)
v = take!(chout)
correct = v.worker == 1 ? correct1 : correct2
for name in fieldnames(CodedComputing.ProfilerOutput)
    @test getfield(correct, name) ≈ getfield(v, name)
end

close(chout)