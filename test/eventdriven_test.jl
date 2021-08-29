using Distributions

nworkers = 3
nwait = 2
comm_distributions = [Uniform(0, 1), Uniform(0, 1), Uniform(0, 1)]
comp_distributions = [Uniform(0, 1), Uniform(10, 11), Uniform(20, 21)]
sim = EventDrivenSimulator(;nwait, nworkers, comm_distributions, comp_distributions)
step!(sim)
@test sim.epoch == 1
@test 10 <= sim.time <= 12
@test sim.isidle == [true, true, false]
@test sim.nfresh == [1, 1, 0]

step!(sim, 3)
@test sim.epoch == 4
@test sim.nfresh == [4, 4, 0]

sim = EventDrivenSimulator(sim)
step!(sim)
@test sim.epoch == 1
@test 10 <= sim.time <= 12
@test sim.isidle == [true, true, false]
@test sim.nfresh == [1, 1, 0]