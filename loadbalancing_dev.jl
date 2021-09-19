using Distributions

# Simulator step! function uses no memory

# optimize! allocates 264.447 MB
# Doubling the time limit roughly doubles memory usage
# Initializing the simulator distributions to actual distributions (as opposed to undef) removes simulation allocation
# After optimizations (primarily moving the internal functions outside) the memory usage for 2s is 256 bytes
# (was about 500 MB)

# Channels allocate the first few times they're used, but then stop allocating
# So let's leave the channels alone for now

# I think the profiler is the main culprit when it comes to memory usage
# Let's check if using locks uses any memory
# Locks use no memory, and seem to be extremely fast
# I could change the profiler-load-balancer system to use locks instead of channels
# The coordinator acquires a lock whenever it has latency data to write
# The latency data is written into a ring-buffer specific for each worker
# The coordinator releases the lock and signals the profiler that there's new data to be processed
# (using another lock, or a semaphore)

# The profiler could be an object instead of a thread
# This object contains a lock and a ring-buffer for each worker
# The coordinator acquires the lock and pushes new samples into the ring-buffers
# The load-balancer acquires the lock and computes the mean and variance for each worker, after which it runs its optimizer
# The load-balancer could also be an object, containing a lock and a partition vector
# The load-balancer writes into this vector whenever it has changes
# At the start of each iteration, the coordinator looks for changes in this vector
# (or it uses some other signaling mechanism to detect changes)

# Channel memory usage seems to be fine
# It allocates whenever it needs to increase its buffer size, but the next time you don't need to re-allocate
# So let's have a look at the profiler memory usage, and let's leave the channels alone for now

# The moving time window allocates
# pushing 100 values allocates 14 KiB

# Let's replace the moving time window with my own
# Yep. Let's replace the moving time window with a ring buffer

function lb_benchmark()
    nworkers = 100
    nwait = round(Int, 100 / 2)
    comp_distributions = [Gamma() for _ in 1:nworkers]
    comm_distributions = [Gamma() for _ in 1:nworkers]
    sim = EventDrivenSimulator(;nwait, nworkers, comm_distributions, comp_distributions)
    @btime step!($sim, 100)
end

function profiler_push(chin)
    for i in 1:100
        timestamp = Time(now())
        v = CodedComputing.ProfilerInput(1, 1/2, 1/10, timestamp, rand(), rand())
        push!(chin, v)
    end    
    return
end

function profiler_take(chout)
    while !isready(chout)
        sleep(0.1)
    end
    while isready(chout)
        take!(chout)
    end
    return
end

function profiler_main()
    # setup the profiler
    # put some values into it
    # take the result out
    # meanwhile, measure the memory usage

    chin, chout = CodedComputing.setup_profiler_channels()
    nworkers = 9
    qlower, qupper = 0.0, 1.0
    windowsize = Second(1)
    minsamples = 1
    task = Threads.@spawn CodedComputing.latency_profiler(chin, chout; nworkers, qlower, qupper, windowsize, minsamples)

    println("pushing")
    profiler_push(chin)
    sleep(0.1)

    println("pushing")
    profiler_push(chin)
    sleep(0.1)
    
    println("pushing")
    profiler_push(chin)
    sleep(0.1)    

    println("taking")
    profiler_take(chout)

    println("pushing")
    profiler_push(chin)
    sleep(0.1)

    println("pushing")
    profiler_push(chin)
    sleep(0.1)
    
    println("pushing")
    profiler_push(chin)
    sleep(0.1)        

    println("taking")
    profiler_take(chout)    

    println("closing")
    close(chin)
    wait(task)
    close(chout)
    return
end

function cb_foo!(w)
    for i in 1:100
        timestamp = Time(now())
        v = CodedComputing.ProfilerInput(i, 1/2, 1/10, timestamp, 1.0, 1.0)
        pushfirst!(w, v)
    end
    w
end

function cb_clear!(w, windowsize)
    t = Time(now())
    rv = 0
    while length(w) > 0 && (t - w[end].timestamp) > windowsize
        pop!(w)
        rv += 1
    end
    rv
end

function cb_main()
    w = CircularBuffer{CodedComputing.ProfilerInput}(100)
    @time cb_foo!(w)
    println(length(w))

    @time var((x)->getfield(x, :comm_delay), w)
    return

    sleep(2)
    # @time cb_foo!(w)
    # println(length(w))
    windowsize = Second(1)
    @time cb_clear!(w, windowsize)
    println(length(w))
end

function window_foo!(w)
    for i in 1:100
        timestamp = Time(now())
        v = CodedComputing.ProfilerInput(i, 1/2, 1/10, timestamp, 1.0, 1.0)
        fit!(w, (timestamp, v))
    end
    w
end

using Dates, OnlineStats
function window_main()
    windowsize = Second(1)
    w = MovingTimeWindow(windowsize, valtype=CodedComputing.ProfilerInput, timetype=Time)
    sizehint!(w.values, 1000)
    # sizehint!(w, 100)
    @time window_foo!(w)
    sleep(1)
    @time window_foo!(w)
    sleep(1)
    @time window_foo!(w)    
    sleep(1)
    @time window_foo!(w)        
end

function lock_main()
    l = ReentrantLock()
    @time lock(l)
    @time unlock(l)
    @time lock(l)
    @time unlock(l)
    @time lock(l)
    @time unlock(l)
end

function ch_foo(ch, n)
    for i in 1:n
        v = Foo(i, i+1, float(i))
        push!(ch, v)
    end
    ch
end

function ch_bar(ch)
    while !isempty(ch)
        take!(ch)
    end
    ch
end

struct Foo
    a::Int
    b::Int
    c::Float64
end

function channel_main()
    ch = Channel{Foo}(Inf)
    # ns = [1000, 500, 2000, 1000]
    for _ in 1:10
        n = ceil(Int, rand() * 100)
        @time ch_foo(ch, n)
        @time ch_bar(ch)
        sleep(0.1)
    end
    sleep(10)
    for _ in 1:10
        n = ceil(Int, rand() * 100)
        @time ch_foo(ch, n)
        @time ch_bar(ch)
        sleep(0.1)
    end    
    # @time ch_foo(ch, 500)
    # @time ch_bar(ch)    
    # @time ch_foo(ch, 2000)
    # @time ch_bar(ch)    
    # @time ch_foo(ch, 1000)
    # @time ch_bar(ch)        
end

function sim_main()

    # PriorityQueue
    # @btime step!($sim, 100)
    # 302.834 μs (1105 allocations: 54.22 KiB)

    # BinaryMinHeap
    # @btime step!($sim, 100)
    # 208.338 μs (1000 allocations: 31.25 KiB)

    # Integer => Int
    # @btime step!($sim, 100)
    # 110.700 μs (0 allocations: 0 bytes)

    nworkers = 9
    nwait = 6
    nsubpartitions = 160.0
    θs = fill(1/nworkers, nworkers)
    ps = fill(nsubpartitions, nworkers)

    # values recorded on eX3 with low variance and very low comm. latency relative to comp.
    comp_mcs = [255.07814185599995, 235.33940168554452, 253.72451957702958, 116.82766857029705, 116.75048824871284, 116.63288112475249, 126.99890071128709, 126.4382596942574, 117.12455760475244]
    comp_vcs = [0.0027557071631392205, 0.003592251480945663, 0.002281988669487589, 0.0007248118462346298, 0.0005408877863892769, 0.0005133790987450482, 0.0005865578758950107, 0.0006378358097673953, 0.0005586345559702721]    
    comm_mcs = [3.373097029702779e-5, 3.189267326732839e-5, 3.427457425742762e-5, 3.50826633663372e-5, 3.664775247524789e-5, 3.760737623762412e-5, 3.5680287128712425e-5, 3.737996039603971e-5, 3.6443900990098125e-5]
    comm_vcs = [1.4755830783262175e-12, 1.5582646160146614e-12, 1.3494372741908877e-12, 2.0233815302431064e-12, 2.7092806020993726e-12, 2.554836195077552e-12, 1.7668490363708934e-12, 1.5867281370452314e-12, 1.8507092575245513e-12]    
   
    comp_distributions = CodedComputing.distribution_from_mean_variance.(Gamma, comp_mcs .* θs ./ ps, comp_vcs .* θs ./ ps)
    comm_distributions = CodedComputing.distribution_from_mean_variance.(Gamma, comm_mcs, comm_vcs)    
    sim = EventDrivenSimulator(;nwait=nwait, nworkers, comm_distributions, comp_distributions)    

    @time step!(sim, 100)
end

function lb_main()
    nworkers = 2
    nwait = 1
    min_processed_fraction = 0.1
    time_limit = 1.0 # must be floating-point
    θs = [0.3, 0.7]
    qs = 1 ./ [2, 3]
    ps = round.(Int, 1 ./ qs)
    chin, chout = CodedComputing.setup_loadbalancer_channels()

    # put some random values into the load-balancer input
    Random.seed!(123)
    worker = 1
    v1 = CodedComputing.ProfilerOutput(worker, θs[worker], qs[worker], rand(), rand(), rand(), rand())
    push!(chin, v1)
    v1 = CodedComputing.ProfilerOutput(worker, θs[worker], qs[worker], rand(), rand(), rand(), rand())
    push!(chin, v1)
    v1 = CodedComputing.ProfilerOutput(worker, θs[worker], qs[worker], rand(), rand(), rand(), rand())
    push!(chin, v1)    

    worker = 2
    v2 = CodedComputing.ProfilerOutput(worker, θs[worker], qs[worker], rand(), rand(), rand(), rand())
    push!(chin, v2)

    # start the load-balancer
    task = Threads.@spawn CodedComputing.load_balancer(chin, chout; min_processed_fraction, nwait, nsubpartitions=ps, nworkers, time_limit, min_improvement=1)
    
    t0 = time_ns()
    while (time_ns() - t0)/1e9 < 10 && isready(chin)
        sleep(0.1)
    end

    while isready(chout)
        take!(chout)
    end

    worker = 1
    v1 = CodedComputing.ProfilerOutput(worker, θs[worker], qs[worker], rand(), rand(), rand(), rand())
    push!(chin, v1)
    v1 = CodedComputing.ProfilerOutput(worker, θs[worker], qs[worker], rand(), rand(), rand(), rand())
    push!(chin, v1)
    v1 = CodedComputing.ProfilerOutput(worker, θs[worker], qs[worker], rand(), rand(), rand(), rand())
    push!(chin, v1)        

    t0 = time_ns()
    while (time_ns() - t0)/1e9 < 10 && isready(chin)
        sleep(0.1)
    end    

    while isready(chout)
        take!(chout)
    end    

    close(chin)
    wait(task)
    close(chout)
    return
end

function main(ps, ps_prev)
    nworkers = 72
    nwait = 36
    # nworkers = 9
    # nwait = 1
    nsubpartitions = 160.0
    # ps = copy(ps0)
    θs = fill(1/nworkers, nworkers)
    # ps = copy(ps_prev)

    # nslow = 3
    # ps_obs = fill(nsubpartitions, nworkers)
    # comp_ms = fill(1.0, nworkers)
    # comp_ms[1:nslow] .*= 2    
    # comp_vs = comp_ms ./ 100
    # comp_mcs = comp_ms ./ (θs ./ ps_obs)
    # comp_vcs = comp_vs ./ (θs ./ ps_obs)


    # comm_mcs = fill(1e-2, nworkers)
    # comm_vcs = comm_mcs ./ 100

    # eX3 values with low variance and very small comm. latency
    # comp_mcs = [255.07814185599995, 235.33940168554452, 253.72451957702958, 116.82766857029705, 116.75048824871284, 116.63288112475249, 126.99890071128709, 126.4382596942574, 117.12455760475244]
    # comp_vcs = [0.0027557071631392205, 0.003592251480945663, 0.002281988669487589, 0.0007248118462346298, 0.0005408877863892769, 0.0005133790987450482, 0.0005865578758950107, 0.0006378358097673953, 0.0005586345559702721]    
    # comm_mcs = [3.373097029702779e-5, 3.189267326732839e-5, 3.427457425742762e-5, 3.50826633663372e-5, 3.664775247524789e-5, 3.760737623762412e-5, 3.5680287128712425e-5, 3.737996039603971e-5, 3.6443900990098125e-5]
    # comm_vcs = [1.4755830783262175e-12, 1.5582646160146614e-12, 1.3494372741908877e-12, 2.0233815302431064e-12, 2.7092806020993726e-12, 2.554836195077552e-12, 1.7668490363708934e-12, 1.5867281370452314e-12, 1.8507092575245513e-12]

    # comm. latency equal to comp. latency
    # comm_mcs = copy(comp_mcs .* θs ./ nsubpartitions)
    # comm_vcs = copy(comp_vcs .* θs ./ nsubpartitions)

    # comm. latency equal to half of comp. latency
    # comm_mcs = copy(comp_mcs .* θs ./ nsubpartitions ./ 2)
    # comm_vcs = copy(comp_vcs .* θs ./ nsubpartitions ./ 2)    

    comp_mcs = [83.83719400727274, 84.56425873454548, 83.24744768000001, 85.09492904727274, 86.62222161454545, 86.56492212363639, 84.18081268363638, 83.2964639418182, 83.63603345454547, 85.52723904, 85.89885661090909, 78.39971525818183, 93.71899182545458, 85.06894481454546, 85.09361448727273, 86.73630545454546, 85.32156596363637, 84.40723630545455, 84.44267933090909, 82.6533564509091, 84.94617053090911, 84.90497064727273, 84.32782423272727, 84.80634338909091, 84.74789800727272, 90.32809192727272, 84.78158516363638, 79.42350749090909, 86.52342138181817, 84.51160738909093, 84.64746606545455, 84.11714967272728, 84.18197172363638, 84.91479365818182, 83.04630289454546, 86.9430754327273, 84.56391953454548, 82.88443828363637, 86.16268706909094, 84.96690315636366, 85.49810461090908, 84.27044596363636, 84.09113727999998, 84.44407127272729, 85.21930240000002, 84.55733469090909, 85.48306798545454, 85.43646283636365, 84.59088482909092, 89.27814429090911, 85.69318813090909, 82.89300008727272, 84.66070068363636, 84.22216052363638, 84.26186385454547, 84.2088694109091, 87.36550842181819, 84.30621387636366, 83.68611566545457, 84.92868846545456, 83.99821399272729, 84.27077440000001, 91.8733388218182, 84.28995083636363, 85.13140084363637, 86.28002333090912, 85.47884549818181, 85.11117451636365, 83.85590574545455, 84.5384607418182, 83.28374976, 84.23413847272728]
    comp_vcs = [0.0011421443791165577, 0.001965937010903701, 0.001080289134005863, 0.0015147010681869218, 0.0019543853740587075, 0.0018525280119243218, 0.0010717424262526945, 0.0013552023930692842, 0.0013218279306052023, 0.001646853834402909, 0.0011971988786598198, 0.0008171234908152827, 0.0011180002317882714, 0.0014849310589688388, 0.0013503309833066961, 0.002003399670388999, 0.002229045828245602, 0.0016310696849770677, 0.0013699264700789177, 0.001313373340831927, 0.001440510953185439, 0.0015190398374354923, 0.0012164160375029052, 0.0008306335177574833, 0.0011758994786901626, 0.0010003010861745945, 0.0013473881742654802, 0.0007348181667276463, 0.0023673479141235, 0.0012086931203448797, 0.0008219273744405475, 0.0012246601989724356, 0.0011504895403956735, 0.0013126575204895218, 0.0010350639226623624, 0.001785326509537193, 0.0013303620153629623, 0.0009981376354634575, 0.0017964488424443016, 0.001630187729733738, 0.0017519484032818982, 0.001787923696131851, 0.0014579672492872793, 0.001153181321664806, 0.0013350055705060714, 0.0011140041528969352, 0.0011822385440088466, 0.001246437813973071, 0.0009866541269556797, 0.0007728917443144192, 0.0014659113197203564, 0.0013141219658085656, 0.0013415545268761183, 0.001504927399968425, 0.000906065910986086, 0.0010870860165416388, 0.0012550196169302695, 0.0008419391399846764, 0.0010460085177690004, 0.0015293742748779598, 0.0009529932101843303, 0.0014254519016226395, 0.0008692140107794807, 0.0008258379861804196, 0.0017921203526416393, 0.0013239507762105058, 0.001927217880471291, 0.0016084685805719378, 0.0008582012743286067, 0.0014071052024702247, 0.0012597131371129017, 0.0013191295808503373]
    comm_mcs = [0.0003480254545454544, 0.0003319341919191919, 0.00026587455555555564, 0.00034138355555555556, 0.00040316833333333324, 0.00039426245454545454, 0.0003415080000000001, 0.0003219556666666666, 0.0003220089191919191, 0.00038495763636363644, 0.0004099111616161616, 0.0002096090606060607, 0.0002259690404040403, 0.00036263934343434346, 0.00039680437373737374, 0.00037868467676767685, 0.00036656339393939405, 0.0003677447070707071, 0.00035856226262626247, 0.0003042973535353536, 0.0003993650808080807, 0.00041439392929292927, 0.00040552959595959587, 0.00040939215151515157, 0.00041946872727272706, 0.0003035104040404039, 0.00041239234343434355, 0.0002445649797979797, 0.0003739147373737374, 0.00042839323232323243, 0.0004370764949494949, 0.00038970366666666657, 0.0004284080202020202, 0.00043861960606060595, 0.00041204406060606067, 0.00039340965656565655, 0.00044712492929292936, 0.0004082409797979797, 0.00039524609090909083, 0.00043600814141414135, 0.0004214690303030302, 0.00044303437373737376, 0.0004123091818181818, 0.000435883898989899, 0.0004593682323232322, 0.0004801075656565655, 0.00043758155555555554, 0.0004502059090909091, 0.0004367451515151514, 0.00026511119191919186, 0.000407111090909091, 0.00044593664646464643, 0.00044228803030303015, 0.000437117595959596, 0.0004675319797979798, 0.0004789977070707069, 0.00035643654545454537, 0.0004665113030303031, 0.00044163787878787893, 0.00040496073737373753, 0.0004637384646464644, 0.0004160973030303031, 0.00020500094949494935, 0.0004884162121212122, 0.0004058337979797981, 0.0003695217171717172, 0.00039872561616161625, 0.00041293092929292917, 0.0004977244747474749, 0.0004509595252525253, 0.00046671063636363655, 0.00047462048484848474]
    comm_vcs = [5.740965332390451e-8, 4.345452720940765e-8, 2.0897826589721686e-8, 3.9728614195903475e-8, 4.783874768185863e-8, 4.746413291931858e-8, 4.628794348258591e-8, 4.656202443670707e-8, 2.7572550612902546e-8, 4.70896132374031e-8, 4.235644613027693e-8, 1.8075039649862175e-9, 8.830231602806428e-9, 3.322531477545785e-8, 5.105769057712298e-8, 3.608274134141073e-8, 3.91950921956125e-8, 4.6208766182308144e-8, 4.083966421352698e-8, 2.4841435527885144e-8, 4.3822700146215706e-8, 4.882312347214655e-8, 4.320883324624073e-8, 3.7677578237199255e-8, 5.280018543405695e-8, 1.924283053094787e-8, 5.917481436683162e-8, 6.488923716141019e-9, 3.19733472787795e-8, 5.665397354431981e-8, 4.2398035149946876e-8, 3.4169793846282844e-8, 4.8652965888928915e-8, 4.50717038498953e-8, 6.389193203478423e-8, 4.8686624690387e-8, 4.9491465989298095e-8, 5.2208408375938995e-8, 3.4346688861133144e-8, 4.8051166583576e-8, 4.31302555697668e-8, 4.834922571694111e-8, 5.623120069057301e-8, 4.2799384903242254e-8, 5.000090975007734e-8, 5.391810740503352e-8, 4.341524975719643e-8, 5.440975129410279e-8, 4.470642305651242e-8, 1.1782371344862134e-8, 4.785839480238568e-8, 6.184400377152143e-8, 5.0141780705342516e-8, 5.53503587409074e-8, 4.327395670434303e-8, 5.5518941104974865e-8, 3.039298803533887e-8, 4.6083321766938514e-8, 4.0201333080389357e-8, 3.993063539980984e-8, 5.314513615560225e-8, 4.481117759217081e-8, 9.494668103913932e-10, 5.5252223501763073e-8, 4.202337081874707e-8, 3.7601071229576554e-8, 4.208200520001426e-8, 3.801127667739909e-8, 5.5409340272451336e-8, 4.8944886495683677e-8, 5.7064170423605156e-8, 5.956194455121953e-8]

    ms = comp_mcs .* θs ./ ps .+ comp_mcs
    p = sortperm(ms, rev=true)
    comp_mcs .= comp_mcs[p]
    comp_vcs .= comp_vcs[p]
    comm_mcs .= comm_mcs[p]
    comm_vcs .= comm_vcs[p]

    # comm_mcs = copy(comp_mcs)
    # comm_vcs = copy(comp_vcs)

    # return CodedComputing.distribution_from_mean_variance.(Gamma, comm_mcs, comm_vcs)
    
    # sim_nwait = ceil(Int, nworkers/2)
    # comp_distributions = Vector{Gamma}(undef, nworkers)
    # comm_distributions = Vector{Gamma}(undef, nworkers)
    comp_distributions = [Gamma() for _ in 1:nworkers]
    comm_distributions = [Gamma() for _ in 1:nworkers]
    sim = EventDrivenSimulator(;nwait, nworkers, comm_distributions, comp_distributions)
    min_processed_fraction = nwait / nworkers / nsubpartitions

    # ls = zeros(nworkers)
    # ∇s = zeros(nworkers)
    contribs = zeros(nworkers)

    ps, latency0, contrib0, latency, contrib = CodedComputing.optimize2!(ps, ps_prev, sim; θs, contribs, comp_mcs, comp_vcs, comm_mcs, comm_vcs, min_contribution=min_processed_fraction, time_limit=1.0)

    @info "ps: $ps"
    @info "contribs: $contribs"    
    ms = comp_mcs .* θs ./ ps .+ comm_mcs
    @info "ms: $ms"
    @info "latency $(latency / latency0), contrib. $(contrib/ contrib0) ($(contrib / min_processed_fraction))"

    if (!isapprox(latency, latency0, rtol=1e-3) && latency < latency0) && (isapprox(contrib, min_processed_fraction, rtol=1e-2) || min_processed_fraction < contrib)
        # latency improved and contribution is high enough
        @info "improved latency; accepting solution"
        ps_prev .= ps
    elseif (!isapprox(contrib, contrib0, rtol=1e-2) && contrib0 < contrib) && ((isapprox(latency, latency0, rtol=1e-3) || latency < latency0))
        # contribution improved and latency remained unchanged or improved
        @info "improved contrib; accepting solution"
        ps_prev .= ps
    else
        @info "no improvement; rejecting solution"
    end

    # if contrib0 < contrib && latency < latency0
    #     @info "accepting new solution"
    #     ps_prev .= ps
    # else
    #     @info "rejecting new solution"    
    # end

    return

    ls = zeros(nworkers)
    simulation_niterations = 100
    simulation_nsamples = 10
    function simulate(ps)

        # setup compute latency distributions
        for i in 1:nworkers
            m = comp_mcs[i] * θs[i] / ps[i]
            v = comp_vcs[i] * θs[i] / ps[i]
            sim.comp_distributions[i] = CodedComputing.distribution_from_mean_variance(Gamma, m, v)
        end
    
        # run nsamples simulations, each consisting of niterations steps
        ls .= 0
        for _ in 1:simulation_nsamples
            sim = EventDrivenSimulator(sim)
            step!(sim, simulation_niterations)
            ls .+= sim.nfresh ./ simulation_niterations
        end
        ls ./= simulation_nsamples
    end

    # setup communication latency distributions
    for i in 1:nworkers
        m = comm_mcs[i]
        v = comm_vcs[i]
        sim.comm_distributions[i] = CodedComputing.distribution_from_mean_variance(Gamma, m, v)
    end

    function finite_diff(ps, i, δ=ceil(Int, ps[i]/10))
        pi0 = ps[i]
        ps[i] = pi0 + δ
        ls = simulate(ps)

        # while isapprox(ls[i], 0)
        #     δ *= 2
        #     ps[i] = pi0 + δ
        #     ls = simulate(1 ./ ps)
        # end

        forward = ls[i] * θs[i] / ps[i]
        ps[i] = pi0 - δ
        ps[i] = max(1, ps[i])
        ls = simulate(ps)
        backward = ls[i] * θs[i] / ps[i]
        rv = (forward - backward) / (pi0 + δ - ps[i])
        ps[i] = pi0
        rv
    end

    function finite_diff1(ps, i, δ=ceil(Int, ps[i]/10))
        ps[1] += δ
        ls = simulate(ps)
        forward = ls[i] * θs[i] / ps[i]
        ps[1] -= 2δ
        ls = simulate(ps)
        backward = ls[i] * θs[i] / ps[i]
        ps[1] += δ       
        println("forward: $forward, backward: $backward")
        (forward - backward) / 2δ
    end

    # println(finite_diff(ps, 1))
    # println(finite_diff(ps, 1))
    # println(finite_diff(ps, 1))

    min_processed_fraction = sim_nwait / nworkers / nsubpartitions

    ls = simulate(ps)
    losses = ls .* θs ./ ps
    f0 = var(losses)
    println("prev. loss: $f0, contribution: $(sum(losses))")
    println(sort(losses))

    Is = collect(1:nworkers)
    for _ in 1:10
        shuffle!(Is)
        # println("start")

        ls = simulate(ps)
        losses = ls .* θs ./ ps
        μ = mean(losses)
        
        for i in Is

            δ = finite_diff(ps, i)

            if isapprox(δ, 0)
                x = ps[i]*1.1
            else
                # solve for x in losses[i] + (x - ps[i]) * δ - μ = 0
                x = (μ - losses[i]) / δ + ps[i]
                x = min(x, ps[i]*1.1)
                x = max(x, ps[i]*0.9)
                x = max(x, 1.0)
            end
            ps[i] = x
            # println("x: $x")

            # solve for x in losses[i] + δ*x - μ = 0
            # x = (μ - losses[i]) / δ
            # ps[i] += 

        end

        # println("finish")

        ls = simulate(ps)        
        losses = ls .* θs ./ ps        
        ps ./= min_processed_fraction / sum(losses)
    end

    ps .= round.(ps)

    ls = simulate(ps)
    losses = ls .* θs ./ ps
    f1 = var(losses)
    # println("$(sum(losses)) / $min_processed_fraction")    
    println("new loss: $f1, contribution: $(sum(losses))")
    println(sort(losses))

    println("f0: $f0, f1: $f1")
    # println(ps)
    if f1 < f0 * 0.99
        println("found a sufficiently better solution")        
        return ps
    else
        println("new solution not sufficiently better; discarding it")
        return ps0
    end
end

"""

Return a lower bound on the log of the probability of `z1 + y1 <= z2 + y2`, where `z1`, `y1`,
`z2`, and `y2` are drawn from the probability distributions `dz1`, `dy1`, `dz2`, and `dy2`,
respectively.
"""
function less_than_lower_bound(dz1, dy1, dz2, dy2)    

    # compute the midpoint of the means
    cz = (mean(dz1) + mean(dz2)) / 2
    cy = (mean(dy1) + mean(dy2)) / 2

    # prob. of z1 <= cz and z2 >= cz
    v1 = logcdf(dz1, cz) + logccdf(dz2, cz)

    # prob. of y1 <= cy and y2 >= cy
    v2 = logcdf(dy1, cy) + logccdf(dy2, cy)

    v1 + v2
end

"""

For the `i`-th random variable, return a lower bound on the log of the probability of a sample drawn 
from this variable being smaller than all other variables.
"""
function less_than_lower_bound(i::Integer, dzs, dys)

    # compute the midpoint of the means
    cz = mean(mean, dzs)
    cy = mean(mean, dys)

    # prob. of zi <= cz and all others >= cz
    pz = logcdf(dzs[i], cz)
    for j in 1:length(dzs)
        if j != i
            pz += logccdf(dzs[j], cz)
        end
    end

    # prob. of yi <= cy and all others >= cy
    py = logcdf(dys[i], cy)
    for j in 1:length(dys)
        if j != i
            py += logccdf(dys[j], cy)
        end
    end

    # prob. of both being true
    pz + py
end

"""

For each random variable, return a lower bound on the log of the probability of a sample drawn 
from this variable being smaller than all other variables.
"""
function less_than_lower_bounds!(rv, dzs, dys)
    length(rv) == length(dzs) || throw(DimensionMismatch("rv has dimension $(length(rv)), but dzs has dimension $(length(dzs))"))
    length(rv) == length(dys) || throw(DimensionMismatch("rv has dimension $(length(rv)), but dys has dimension $(length(dys))"))
    n = length(rv)
    rv .= 0
    
    # compute the midpoint of the means
    cz = mean(mean, dzs)
    cy = mean(mean, dys)    

    # for each i, prob. of zi <= cz and all others >= cz
    pz = 0.0
    for i in 1:n
        pz += logccdf(dzs[i], cz)
    end
    for i in 1:n
        v = logccdf(dzs[i], cz)
        pz -= v
        rv[i] += pz + logcdf(dzs[i], cz)
        pz += v
    end

    # for each i, prob. of yi <= cy and all others >= cy
    py = 0.0
    for i in 1:n
        py += logccdf(dys[i], cy)
    end
    for i in 1:n
        v = logccdf(dys[i], cy)
        py -= v
        rv[i] += py + logcdf(dys[i], cy)
        py += v
    end

    rv
end

less_than_lower_bounds(dzs, dys) = less_than_lower_bounds!(zeros(length(dzs)), dzs, dys)

# The lower bound looks great
# Next, let's change the load-balancer to evaluate contribution in the log domain

function bound_main()




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

    println([less_than_lower_bound(i, dzs, dys) for i in 1:nworkers])
    println(less_than_lower_bounds(dzs, dys))
    return

    dz1 = Gamma(1)
    dz2 = Gamma(1)
    dy1 = Gamma(2)
    dy2 = Gamma(1)

    # bound prob. of 1 being faster than 2

    # take the midpoint of the means
    cz = (mean(dz1) + mean(dz2)) / 2
    cy = (mean(dy1) + mean(dy2)) / 2

    @info "cz: $cz, cy: $cy"

    # prob. of z1 <= cz and z2 >= cz
    v1 = cdf(dz1, cz) * (1 - cdf(dz2, cz))

    # prob. of y1 <= cy and y2 >= cy
    v2 = cdf(dy1, cy) * (1 - cdf(dy2, cy))

    dzs = [dz1, dz2]
    dys = [dy1, dy2]
    v1 * v2, exp(less_than_lower_bound(dz1, dy1, dz2, dy2)), exp(less_than_lower_bound(1, dzs, dys))
end