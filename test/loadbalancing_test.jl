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