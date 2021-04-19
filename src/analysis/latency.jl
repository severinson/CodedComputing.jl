# Code for analyzing and plotting latency

# linear model
# get_βs() = [0.005055059937837611, 8.075122937312302e-8, 1.1438758464435006e-16]
# get_γs() = [0.03725188744901591, 3.109510011653974e-8, 6.399147477943208e-16]

# get_offset(w) = 0.005055059937837611 .+ 8.075122937312302e-8w .+ 1.1438758464435006e-16w.^2
# get_slope(w, nworkers) = 0.03725188744901591 .+ 3.109510011653974e-8(w./nworkers) .+ 6.399147477943208e-16(w./nworkers).^2

# get_offset(w) = 7.927948909471475e-9w
# get_slope(w, nworkers) = (-0.7293548623181831 .+ 0.046797154093631756log(c))./nworkers

# linear model based on fully shuffled data
get_offset(c) = 7.517223358654102e-9c
get_slope(c, nworkers) = (0.0034845934661738285 .+ 1.6086658767640374e-9c) ./ nworkers

# shifted exponential model
# get_shift(w) = 0.2514516116132241 .+ 6.687583396247953e-8w .+ 2.0095825408761404e-16w.^2
# get_scale(w) = 0.23361469930191084 .+ 7.2464826067975726e-9w .+ 5.370433628859458e-17w^2

get_shift(w) = 7.511478910765988e-9w
get_scale(w) = 0.012667298954788542 .+ 5.788848515637547e-10w

"""

Fit a shifted exponential latency model to the data.
"""
function fit_shiftexp_model(df, worker_flops)
    # df = df[df.nwait .== nwait, :]
    df = df[isapprox.(df.worker_flops, worker_flops, rtol=1e-2), :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.kickstart .== false, :]
    if size(df[df.nwait .==1, :], 1) == 0
        return NaN, NaN
    end

    # get the shift from waiting for 1 worker
    shift = quantile(df[df.nwait .== 1, :latency], 0.01)
    ts = df.latency .- shift

    # get the scale from waiting for all workers
    β = 0.0
    for nworkers in unique(df.nworkers)
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        nwait = nworkers
        ts = dfi[dfi.nwait .== nwait, :latency] .- shift
        # σ = var(ts)
        # β1 = sqrt(σ / sum(1/i^2 for i in (nworkers-nwait+1):nworkers))        
        μ = mean(ts)
        βi = μ / sum(1/i for i in (nworkers-nwait+1):nworkers)
        β += βi * size(dfi, 1) / size(df, 1)
    end
    return shift, β
end

"""

Plot the shifted exponential shift and scale as a function of w.
"""
function plot_shiftexp_model(df)
    ws = sort!(unique(df.worker_flops))
    models = [fit_shiftexp_model(df, w) for w in ws]
    shifts = [m[1] for m in models]
    scales = [m[2] for m in models]

    # filter out nans
    mask = findall(.!isnan, scales)
    ws = ws[mask]
    shifts = shifts[mask]
    scales = scales[mask]

    plt.figure()
    plt.plot(ws, shifts, "o")

    poly = Polynomials.fit(ws, shifts, 1)
    println(poly.coeffs)    
    ts = range(0, maximum(df.worker_flops), length=100)
    plt.plot(ts, poly.(ts))

    plt.grid()
    plt.xlabel("w")    
    plt.ylabel("shift")

    plt.figure()
    plt.plot(ws, scales, "o")

    poly = Polynomials.fit(ws, scales, 1)
    println(poly.coeffs)
    ts = range(0, maximum(df.worker_flops), length=100)
    plt.plot(ts, poly.(ts))    

    plt.grid()
    plt.xlabel("w")    
    plt.ylabel("scale")    

    return
end

"""

Return the number of workers that minimizes t_compute, when the workload is `σ`,
and the coordinator waits for the fastest `f` fraction of workers.
"""
function optimize_nworkers(σ0, f)
    βs = get_βs()
    γs = get_γs()
    c1 = γs[1]*f
    c2 = (βs[2] + γs[2]*f)*σ0
    sqrt(c2) / sqrt(c1)    
end

"""

Plot latency as a function of `nworkers` for a fixed total workload.
"""
function plot_predictions(c0=1.6362946777247114e9; df=nothing, dfo=nothing)
    if !isnothing(dfo)
        dfo = dfo[dfo.order .<= dfo.nwait, :]
    end
    nworkers = 1:500    
    c = c0 ./ nworkers
    plt.figure()

    for psi in [1/12, 0.5, 1.0]
        nwait = psi.*nworkers
        # return nworkers, predict_latency.(c, nwait, nworkers)
        plt.plot(nworkers, predict_latency.(c, nwait, nworkers), label="$psi")

        if !isnothing(df)
            dfi = df
            dfi = dfi[dfi.nwait .== round.(Int, psi.*dfi.nworkers), :]
            dfi = dfi[isapprox.(dfi.worker_flops .* dfi.nworkers, c0, rtol=1e-2), :]
            if size(dfi, 1) > 0
                # plt.plot(dfo.nworkers, dfo.worker_latency, ".") # all points
                dfj = by(dfi, :nworkers, :latency => mean => :latency)
                plt.plot(dfj.nworkers, dfj.latency, "s", label="psi: $psi (df)") # averages
            end
        end

        if !isnothing(dfo)
            dfi = dfo
            dfi = dfi[dfi.order .== round.(Int, psi.*dfi.nworkers), :]
            dfi = dfi[isapprox.(dfi.worker_flops .* dfi.nworkers, c0, rtol=1e-2), :]        
            if size(dfi, 1) > 0            
                # plt.plot(dfo.nworkers, dfo.worker_latency, ".") # all points
                dfj = by(dfi, :nworkers, :worker_latency => mean => :worker_latency)
                plt.plot(dfj.nworkers, dfj.worker_latency, "o", label="psi: $psi (dfo)") # averages
            end
        end        
    end

    plt.legend()
    plt.xlabel("nworkers")
    plt.ylabel("Latency [s]")
    plt.grid()
    return
end

"""

Plot latency as a function of nworkers for some value of σ0
σ0=1.393905852e9 is the workload associated with processing all data on 1 worker
"""
function plot_predictions_old(σ0=1.393905852e9; df=nothing)

    nworkers_all = 1:50
    σ0s = 10.0.^range(5, 12, length=20)    

    # plot the speedup due to waiting for fewer workers    
    for fi in [0.1, 0.5]
        f1 = 1.0
        f2 = fi
        nws1 = optimize_nworkers.(σ0s, f1)
        nws2 = optimize_nworkers.(σ0s, f2)
        ts1 = get_offset.(σ0s./nws1) .+ get_slope.(σ0s./nws1, nws1) .* f1 .* nws1
        ts2 = get_offset.(σ0s./nws2) .+ get_slope.(σ0s./nws2, nws2) .* f2 .* nws2
        plt.semilogx(σ0s, ts2 ./ ts1, label="f: $fi")
    end
    plt.xlabel("σ0")
    plt.ylabel("speedup")
    plt.grid()
    plt.legend()          

    # plot the optimized t_compute as a function of σ0
    plt.figure()    
    for f in [0.1, 0.5, 1.0]
        nws = optimize_nworkers.(σ0s, f)
        ts = get_offset.(σ0s ./ nws) .+ get_slope.(σ0s./nws, nws) .* f .* nws
        plt.loglog(σ0s, ts, label="f: $f")

        # print values
        # println("f: $f")
        # for i in 1:length(nws)
        #     println("$(σ0s[i]) $(ts[i])")
        # end                
    end
    # plt.ylim(0, 10)
    plt.xlabel("σ0")
    plt.ylabel("T_compute*")
    plt.grid()
    plt.legend()
    # return
    
    # plot the optimized number of workers as a function of σ0
    plt.figure()
    for f in [0.1, 0.5, 1.0]
        nws = optimize_nworkers.(σ0s, f)
        plt.loglog(σ0s, nws, label="f: $f")

        # print values
        # println("f: $f")
        # for i in 1:length(nws)
        #     println("$(σ0s[i]) $(nws[i])")
        # end        
    end
    # plt.ylim(0, 10)
    plt.xlabel("σ0")
    plt.ylabel("Nn*")
    plt.grid()
    plt.legend()    

    # fix total amount of work    
    plt.figure()        
    for nsubpartitions in [1, 3, 20]
        f = 1.0
        npartitions = nworkers_all .* nsubpartitions
        ts = get_offset.(σ0 ./ npartitions) .+ get_slope.(σ0 ./ npartitions, nworkers_all) .* f .* nworkers_all
        plt.plot(nworkers_all, ts, label="Np: $nsubpartitions")

        # println("nsubpartitions: $nsubpartitions")
        # for i in 1:length(ts)
        #     println("$(nworkers_all[i]) $(ts[i])")
        # end

        # # point at which the derivative with respect to nworkers is zero
        σ = σ0/nsubpartitions
        x = optimize_nworkers(σ, f)

        if x <= length(ts)
            plt.plot([x], ts[round(Int, x)], "o")
            println("Np: $nsubpartitions, x: $x, t: $(ts[round(Int, x)])")        
        end

        f = 0.5
        npartitions = nworkers_all .* nsubpartitions
        ts = get_offset.(σ0 ./ npartitions) .+ get_slope.(σ0 ./ npartitions, nworkers_all) .* f .* nworkers_all
        plt.plot(nworkers_all, ts, "--", label="Np: $nsubpartitions (1/2)")

        println("nsubpartitions: $nsubpartitions")
        for i in 1:length(ts)
            println("$(nworkers_all[i]) $(ts[i])")
        end

        # # point at which the derivative with respect to nworkers is zero
        σ = σ0/nsubpartitions
        x = optimize_nworkers(σ, f)

        if x <= length(ts)
            plt.plot([x], ts[round(Int, x)], "o")
            println("Np: $nsubpartitions, x: $x, t: $(ts[round(Int, x)])")        
        end                
    end

    # Np = 3
    f = 1.0
    dfi = df
    dfi = dfi[dfi.kickstart .== false, :]
    dfi = dfi[dfi.nreplicas .== 1, :]
    dfi = dfi[dfi.pfraction .== 1, :]
    dfi = dfi[dfi.nsubpartitions .== 3, :]
    dfi = dfi[dfi.nwait .== round.(Int, f.*dfi.nworkers), :]
    dfj = combine(groupby(dfi, :nworkers), :t_compute => mean)
    plt.plot(dfj.nworkers, dfj.t_compute_mean, "s")
    plt.plot(dfi.nworkers, dfi.t_compute, ".")
    for i in 1:size(dfj, 1)
        println("$(dfj.nworkers[i]) $(dfj.t_compute_mean[i])")
    end
    println()
    println((minimum(dfi.worker_flops.*dfi.nworkers), maximum(dfi.worker_flops.*dfi.nworkers)))

    # expression for α1 + α2*(f*nworkers)
    # (to make sure it's correct)
    # ts = [βs[1] + γs[1]*f*nworkers + (βs[2]+γs[2]*f)*σ0/nworkers + βs[3]*(σ0/nworkers)^2 + γs[3]*f*σ0^2/nworkers^3 for nworkers in nworkers_all]
    # plt.plot(nworkers_all, ts, "--")

    plt.ylim(0, 10)
    plt.xlabel("Nn")
    plt.ylabel("T_compute [s]")
    plt.title("Fix total amount of work")    
    plt.grid()
    plt.legend()        
    return
end

"""

Plot the CCDF of the iteration time for all values of `nwait` for the given number of workers.
"""
function plot_iterationtime_cdf(df; nworkers::Integer=12)
    df = df[df.nworkers .== nworkers, :]
    plt.figure()
    for nwait in sort!(unique(df.nwait))
        df_nwait = df[df.nwait .== nwait, :]
        x = sort(df_nwait.t_compute)
        y = 1 .- range(0, 1, length=length(x))
        plt.semilogy(x, y, label="($nworkers, $nwait")
    end
    plt.ylim(1e-2, 1)
    plt.xlabel("Iteration time [s]")
    plt.ylabel("CCDF")
    plt.grid()
    plt.legend()
    plt.show()
end

function deg3_model_dfo(dfo)
    dfo = dfo[dfo.nwait .== dfo.nworkers, :]
    rv = by(
        dfo, [:nworkers, :worker_flops],
        [:order, :worker_latency] => ((x) -> NamedTuple{(:x1, :x2, :x3, :x4)}(fit_polynomial(x.order, x.worker_latency, 3)[2])),
    )
    rv.x3n = -1 .* rv.x3
    sort!(rv, [:nworkers, :worker_flops])
    rv   
end

function fit_deg3_model(dfo)
    dfo = dfo[dfo.order .<= dfo.nwait, :]
    A = zeros(size(dfo, 1), 8)
    A[:, 1] .= 1
    A[:, 2] .= dfo.order
    A[:, 3] .= dfo.order.^2
    A[:, 4] .= dfo.order.^3
    A[:, 5] .= dfo.worker_flops
    A[:, 6] .= dfo.worker_flops .* dfo.order ./ dfo.nworkers
    A[:, 7] .= dfo.worker_flops .* (dfo.order ./ dfo.nworkers).^2
    A[:, 8] .= dfo.worker_flops .* (dfo.order ./ dfo.nworkers).^3
    y = dfo.worker_latency
    mask = .!isinf.(y)
    x = A[mask, :] \ y[mask]
    for (i, label) in enumerate(["b1", "c1", "d1", "e1", "b2", "c2", "d2", "e2"])
        println("$label = $(x[i])")
    end
    x
end

function deg3_coeffs(type="c5xlarge")
    if type == "c5xlarge"
        # b1 = -0.0005487338276092924
        b1 = 0
        c1 = 0.00011666153003402824
        d1 = -2.200065092782715e-6
        e1 = 1.3139560334678954e-8
        b2 = 7.632075760960183e-9
        c2 = 2.1903320927807077e-9
        d2 = -4.525831193535335e-9
        e2 = 4.336744075595763e-9
        return b1, c1, d1, e1, b2, c2, d2, e2
    elseif type == "t3large"
        b1 = -0.0012538429018191268
        c1 = 5.688267095613402e-5
        d1 = 1.8724136277744778e-6
        e1 = -1.2889725208620691e-8
        b2 = 8.140573448894689e-9
        c2 = 5.388607340950452e-9
        d2 = -1.1648036394321019e-8
        e2 = 7.880211300623262e-9
        return b1, c1, d1, e1, b2, c2, d2, e2
    end    
    error("no instance type $type")
end

function predict_latency(c, nwait, nworkers; type="c5xlarge")
    b1, c1, d1, e1, b2, c2, d2, e2 = deg3_coeffs(type)
    rv = b1 + b2*c
    rv += c1*nwait + c2*c*nwait/nworkers
    rv += d1*nwait^2 + d2*c*(nwait/nworkers)^2
    rv += e1*nwait^3 + e2*c*(nwait/nworkers)^3
    rv
end

function plot_deg3_model(dfm)
    plt.figure()
    cols = [:x1, :x2, :x3n, :x4]    
    for col in cols
        dfm = dfm[dfm[col] .> 0, :]
    end
    b1, c1, d1, e1, b2, c2, d2, e2 = deg3_coeffs()        
    for (i, col) in enumerate(cols)
        plt.subplot(2, 2, i)
        plt.title("$col")
        p = i-1
        for nworkers in sort!(unique(dfm.nworkers))
            dfi = dfm
            dfi = dfi[dfi.nworkers .== nworkers, :]
            xs = dfi.worker_flops ./ nworkers.^p
            ys = dfi[col]
            plt.plot(xs, ys, "o", label="N: $nworkers")
        end

        xs = dfm.worker_flops ./ dfm.nworkers.^p
        ys = dfm[col]

        # if col == :x1
        #     intercept = 0
        # else
        #     intercept = 0.1.*minimum(ys)
        # end
        # slope = mean((ys.-intercept) ./ xs)


        if col == :x1
            intercept, slope = b1, b2
        elseif col == :x2
            intercept, slope = c1, c2
        elseif col == :x3n
            intercept, slope = -1*d1, -1*d2
        elseif col == :x4
            intercept, slope = e1, e2
        end
        coeffs = [intercept, slope]        
        
        ts = range(minimum(xs), maximum(xs), length=100)                
        plt.plot(ts, intercept.+slope.*ts)        
        println("$col: $coeffs")

        plt.xscale("log")
        plt.yscale("log")

        plt.tight_layout()
        plt.legend()
        plt.grid()
        plt.xlabel("nflops / nworkers^($p)")
        plt.ylabel("$col")
    end
    return
end

"""

Fit a line to the linear-looking middle part of the orderstats plot.
"""
function linear_model_df(df)
    df = df[df.nwait .<= 0.75.*df.nworkers, :]
    df = df[df.nwait .>= 0.05.*df.nworkers, :]
    rv = by(
        df, [:nworkers, :worker_flops],
        [:nwait, :latency] => ((x) -> NamedTuple{(:intercept, :slope)}(fit_polynomial(x.nwait, x.latency, 1)[2])),
    )
    for nworkers in unique(rv.nworkers)
        row = Dict(:nworkers=>nworkers, :worker_flops=>0, :intercept=>0, :slope=>0)
        push!(rv, row)
    end
    sort!(rv, [:nworkers, :worker_flops])    
    rv
end

"""

Fit a line to the linear-looking middle part of the orderstats plot.
"""
function linear_model_dfo(dfo; onlyedges=true)
    dfo = dfo[dfo.nwait .== dfo.nworkers, :]
    if onlyedges
        dfo = dfo[(dfo.order .== 1) .| (dfo.order .== dfo.nworkers), :]
    end
    # dfo = dfo[dfo.order .<= 0.95.*dfo.nworkers, :]
    # dfo = dfo[dfo.order .>= 0.05.*dfo.nworkers, :]
    rv = by(
        dfo, [:nworkers, :worker_flops],
        [:order, :worker_latency] => ((x) -> NamedTuple{(:intercept, :slope)}(fit_polynomial(x.order, x.worker_latency, 1)[2])),
    )
    sort!(rv, [:nworkers, :worker_flops])
    rv    
end

"""

Return a DataFrame composed of the mean latency for each job.
"""
function mean_latency_df(df)
    df = by(df, :jobid, :latency => mean => :latency, :nworkers => mean => :nworkers, :nwait => mean => :nwait, :worker_flops => mean => :worker_flops)
    df.worker_flops .= round.(df.worker_flops, digits=6) # avoid rounding errors
    df
end

"""

Same as linear_model_df, except that the linear model parameters are fit to the average latency for each experiment.
"""
function mean_linear_model_df(df)
    # df = df[df.kickstart .== false, :]
    # df = df[df.nreplicas .== 1, :]
    df = mean_latency_df(df)
    df = by(df, [:nworkers, :worker_flops], [:nwait, :latency] => (x) -> NamedTuple{(:intercept, :slope)}(fit_polynomial(x.nwait, x.latency, 1)[2]))
    sort!(df, [:nworkers, :worker_flops])
end

"""

Plot the linear model parameters 
"""
function plot_linear_model(df, dfo, dfm=nothing)
    # df = df[df.kickstart .== false, :]
    # df = df[df.nreplicas .== 1, :]
    # df = df[df.pfraction .== 1, :]

    # df = df[df.worker_flops .< 1e9, :]
    # dfo = dfo[dfo.worker_flops .< 1e9, :]

    if isnothing(dfm)
        dfm = linear_model_dfo(dfo)
        # dfm = linear_model_df(df)
        # dfm = mean_linear_model_df(df)
        # dfm = order_linear_model_df(df)
    end

    # dfm = dfm[dfm.worker_flops .< 1e9, :]    
    dfm = dfm[dfm.nworkers .> 3, :]

    # offset
    plt.figure()
    for nworkers in unique(dfm.nworkers)
        if nworkers < 3
            continue
        end
        dfi = dfm
        dfi = dfi[dfi.nworkers .== nworkers, :]        
        dfi = dfi[dfi.worker_flops .> 0, :]
        plt.loglog(dfi.worker_flops, dfi.intercept, ".", label="Nn: $nworkers")

        # print parameters
        # println("Nn: $nworkers")
        # sort!(dfi, [:worker_flops])
        # for i in 1:size(dfi, 1)
        #     println("$(dfi.worker_flops[i]) $(dfi.intercept[i])")
        # end
    end

    # # fitted line
    # poly, coeffs = fit_polynomial(float.(dfm.worker_flops), float.(dfm.intercept), 1)    
    # # t = range(0, maximum(dfm.worker_flops), length=100)
    # ts = exp.(range(log(1), log(maximum(dfm.worker_flops)), length=100))
    # plt.loglog(ts, poly.(ts))   
    
    # fitted line with intercept 0
    dfi = dfm[dfm.worker_flops .> 0, :]
    xs = dfi.worker_flops
    ys = dfi.intercept
    slope = mean(ys ./ xs)
    println("slope: $slope")
    ts = exp.(range(log(1), log(maximum(xs)), length=200))
    plt.loglog(ts, ts.*slope) 

    # # print fit line
    # println("Intercept: ", coeffs)
    # # for i in 1:length(t)
    # #     println("$(t[i]) $(poly(t[i]))")
    # # end

    plt.xlim(0)
    plt.ylim(0)
    plt.grid()
    plt.xlabel("Flops")
    plt.ylabel("Intercept")
    # plt.legend()

    # slope
    plt.figure()

    # fitted line with intercept 0
    dfm = dfm[dfm.slope .> 0, :]    
    xs = dfm.worker_flops ./ dfm.nworkers
    ys = dfm.slope
    slope = mean(ys ./ xs)
    println("slope: $slope")
    ts = exp.(range(log(1), log(maximum(xs)), length=200))
    plt.loglog(ts, ts.*slope, "k--")

    # fitted line with guessed 0
    xs = dfm.worker_flops ./ dfm.nworkers
    ys = dfm.slope
    intercept = 0.5*minimum(ys)
    slope = mean((ys.-intercept) ./ xs)
    println("slope: $slope")
    ts = exp.(range(log(1), log(maximum(xs)), length=200))
    plt.loglog(ts, intercept.+ts.*slope, "m--")

    for nworkers in unique(dfm.nworkers)
        if nworkers < 3
            continue
        end
        dfi = dfm
        dfi = dfi[dfi.nworkers .== nworkers, :]
        xs = dfi.worker_flops ./ nworkers
        # xs = dfi.worker_flops
        ys = dfi.slope
        plt.loglog(xs, ys, ".", label="Nn: $nworkers")

        # print parameters
        # println("Nn: $nworkers")
        # sort!(dfi, [:worker_flops])
        # for i in 1:size(dfi, 1)
        #     println("$(x[i]) $(dfi.slope[i])")
        # end        
    end

    # fitted line
    dfi = dfm[dfm.slope .> 0, :]
    xs = float.(dfi.worker_flops ./ dfi.nworkers)
    ys = float.(dfi.slope)
    # poly, coeffs = fit_polynomial(float.(dfm.worker_flops), float.(dfm.slope .* dfm.nworkers), 1)
    # poly, coeffs = fit_polynomial(xs, ys, 2)
    poly = Polynomials.fit(xs, ys, 1)
    # t = range(0, maximum(log.(dfm.worker_flops)), length=100)
    # t = range(0, maximum(dfm.worker_flops), length=100)
    ts = exp.(range(log(1), log(maximum(xs)), length=200))
    plt.loglog(ts, poly.(ts))
    println(poly.coeffs)

    plt.grid()
    plt.xlabel("Flops / nworkers")    
    plt.ylabel("Slope")
    # plt.xlabel("Flops")    
    # plt.ylabel("Slope x Total number of workers")
    plt.legend()
    plt.xlim(0)
    plt.ylim(0)        
    return
end

function plot_latency_flops(df)
    plt.figure()
    df = df[df.nwait .== df.nworkers, :]
    for nworkers in sort!(unique(df.nworkers))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        plt.plot(dfi.worker_flops .* dfi.nworkers, dfi.latency, ".", label="N: $nworkers")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Total number of flops per iteration")
    plt.ylabel("Latency[s]")
    return
end

"""

Plots:
- Empirical latency (samples and sample average)
- Latency predicted by the proposed linear model
- Latency predicted by the i.i.d. shifted exponential model

"""
function plot_latency(df, nworkers=36)
    df = df[df.kickstart .== false, :]
    df = df[df.nreplicas .== 1, :]
    df = df[df.nworkers .== nworkers, :]
    # df = mean_latency_df(df)

    plt.figure()
    for worker_flops in sort!(unique(df.worker_flops), rev=true)
        dfi = df
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        println(mean(dfi.nsubpartitions))

        # scatter plot of the samples
        plt.plot(dfi.nwait, dfi.latency, ".", label="c: $(round(worker_flops, sigdigits=3))")    
        # write_table(dfi.nwait, dfi.t_compute, "./data/model_raw.csv")

        # compute average delay and quantiles
        dfj = combine(
            groupby(dfi, :nwait), 
            :latency => mean => :latency,
            :latency => ((x)->quantile(x, 0.1)) => :q1,
            :latency => ((x)->quantile(x, 0.9)) => :q9,
            )
        sort!(dfj, :nwait)
        plt.plot(dfj.nwait, dfj.latency, "o")
        # write_table(dfj.nwait, dfj.t_compute_mean, "./data/model_means.csv")    

        # println("Latency average flops: $worker_flops")
        # for i in 1:size(dfj, 1)
        #     println("$(dfj[i, :nwait]) $(dfj[i, :t_compute_mean])")
        # end

        # # plot predicted delay (by a local model)
        # poly, coeffs = fit_polynomial(float.(dfi.nwait), float.(dfi.t_compute), 1)
        # xs = [0, nworkers]
        # ys = poly.(xs)
        # plt.plot(xs, ys, "--")            
        # println(poly)

        # # print values
        # println("local model")
        # for i in 1:length(xs)
        #     println("$(xs[i]) $(ys[i])")
        # end

        # plot predicted delay (by the global model)
        println("c: $worker_flops")
        xs = [0, nworkers]
        ys = get_offset(worker_flops) .+ get_slope(worker_flops, nworkers) .* xs
        plt.plot(xs, ys)
        # write_table(xs, ys, "./data/model_linear.csv")
        println("global model")
        for i in 1:length(xs)
            println("$(xs[i]) $(ys[i])")
        end


        # plot delay predicted by the shifted exponential order statistics model
        # shift, β = fit_shiftexp_model(df, w)
        shift = get_shift(worker_flops)
        scale = get_scale(worker_flops)
        ys = [mean(ExponentialOrder(scale, nworkers, nwait)) for nwait in 1:nworkers] .+ shift
        plt.plot(1:nworkers, ys, "--")
        # # write_table(xs, ys, "./data/model_shiftexp.csv")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("w")
    plt.ylabel("Latency [s]")
    return
end

"""

DataFrame composed of linear model parameters fit the order statistics latency.
"""
function order_model_df(df)
    df = df[df.nwait .== df.nworkers, :]
    df = orderstats_df(df)
    df = df[df.isstraggler .== false, :]    
    df = by(df, [:nworkers, :worker_flops], [:order, :worker_latency] => (x) -> NamedTuple{(:intercept, :x1, :x2, :x3)}(fit_polynomial(x.order, x.latency, 3)[end]))
    sort!(df, [:nworkers, :worker_flops])
end

function plot_order_model(df)
    df = order_model_df(df)
    
    for nworkers in sort!(unique(df.nworkers))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        plt.plot(dfi.worker_flops, dfi.intercept, ".", label="Nn: $nworkers")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("flops")
    plt.ylabel("Intercept")
    return
end

"""

Plot the latency of the w-th fastest worker as a function of worker_flops for all nworkers.
"""
function plot_order_latency(df, order=1)
    df = df[df.nwait .== df.nworkers, :]
    df = orderstats_df(df)
    df = df[df.isstraggler .== false, :]
    # df = df[df.order .== df.nworkers, :] # TODO: fix, not order 
    for nworkers in sort!(unique(df.nworkers))
        dfi = df
        dfi = dfi[dfi.nworkers .== nworkers, :]
        # plt.plot(dfi.worker_flops, dfi.latency, ".", label="Nn: $nworkers")
        dfj = by(dfi, :worker_flops, :worker_latency => mean)
        plt.plot(dfj.worker_flops, dfj.latency_mean ./ nworkers, "o", label="Nn: $nworkers")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("flops")
    plt.ylabel("Latency [s]")
    return
end

function order_test()
    plt.figure()
    for total in [9, 18, 36, 72]
        ys = [mean(ExponentialOrder(1.0, total, i)) for i in 1:total]
        # xs = (1:total) ./ total
        xs = 1:total
        plt.plot(xs, ys, "-o", label="$total")
    end
    plt.grid()
    plt.legend()
    return
end

"""

Plot order statistics latency for a given computational load.

Plot
- Latency order stats recorded individually for each worker for different w_target
- Iteration latency for different w_target
"""
function plot_orderstats(dfo; worker_flops=1.08e7, onlycompute=false, normalized=false)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    dfo = dfo[dfo[order_col] .<= dfo.nwait, :]    
    dfo = dfo[isapprox.(dfo.worker_flops, worker_flops, rtol=0.01), :]
    println("worker_flops:\t$(unique(dfo.worker_flops))")
    println("nbytes:\t$(unique(dfo.nbytes))\n")

    # fit a linear model to the overall latency
    # dfm = linear_model_dfo(dfo)

    plt.figure()

    # plot latency of individual workers
    colors = ["b", "g", "r", "c", "m", "y"]
    for nworkers in sort!(unique(dfo.nworkers))
        dfi = dfo
        dfi = dfi[dfi.nworkers .== nworkers, :]

        # # plot overall latency        
        # dfj = by(dfi, [:nworkers, :nwait], :latency => mean => :latency)
        # sort!(dfj, :nwait)
        # plt.plot(dfj.nwait, dfj.latency, "k-", label="Overall latency")        

        # # plot a linear model fit to the overal latency
        # dfj = dfm[dfm.nworkers .== nworkers, :]
        # dfj = dfj[dfj.worker_flops .> 0, :]

        # @assert size(dfj, 1) == 1
        # intercept = dfj.intercept[1]
        # slope = dfj.slope[1]
        # ts = [0, 36]
        # plt.plot(ts, intercept.+ts.*slope, "k--")

        xs = zeros(0)
        ys = zeros(0)
        for (color, nwait) in zip(colors, sort!(unique(dfi.nwait), rev=true))
            if nwait != nworkers
                continue
            end
            dfj = dfi
            dfj = dfj[dfj.nwait .== nwait, :]

            # all samples
            # plt.plot(dfj[order_col] ./ dfj.nworkers, dfj[latency_col], color*".")
            
            # mean latency
            dfk = by(dfj, order_col, latency_col => mean => :mean)
            sort!(dfk, order_col)
            if normalized
                xs = dfk[order_col] ./ nworkers
                ys = dfk.mean
            else
                xs = dfk[order_col]
                ys = dfk.mean
            end
            plt.plot(xs, ys, "-o", label="N_n: $nworkers")
            write_table(xs, ys, "orderstats_$(nworkers)_empirical.csv")

            # store the overall iteration latency
            sort!(dfk, order_col)
            push!(xs, nwait)
            push!(ys, dfk.mean[end])

            # fit a polynomial to the data
            p, coeffs = fit_polynomial(xs, ys, 3)
            println("N: $nworkers, $coeffs")
            # p, coeffs = fit_polynomial(xs[[1, length(xs)]], ys[[1, length(ys)]], 1)
            xs = range(0, maximum(xs), length=100)
            ys = p.(xs)
            plt.plot(xs, ys, "k--")
            write_table(xs, ys, "orderstats_$(nworkers)_local.csv")

            # plot predicted latency
            xs = 1:nwait
            ys = predict_latency.(worker_flops, 1:nwait, nworkers, type="c5xlarge")
            plt.plot(xs, ys, "m--")
            write_table(xs, ys, "orderstats_$(nworkers)_local.csv")
        end

        # plot the overall iteration latency
        # plt.plot(xs, ys, "k-")

        # fit a line to the linear-looking middle part
        # mask = (nworkers * 0.05) .<= xs .<= (nworkers * 0.75)
        # p, coeffs = fit_polynomial(xs[mask], ys[mask], 1)
        # ts = [0, nworkers]
        # plt.plot(ts, p.(ts), "k-")

        # simulated latency
        # ys = [simulate_orderstats(1000, 100, nworkers, i) for i in 1:nworkers]
        # plt.plot((1:nworkers), ys, label="Simulated ($nworkers workers)")
    end
    if normalized
        plt.xlim(0, 1)    
        plt.xlabel("w / Total number of workers")        
    else
        plt.xlim(0)    
        plt.xlabel("w")        
    end
    plt.legend()
    plt.ylabel("Latency [s]")
    plt.tight_layout()
    plt.grid()
    return
end

function plot_orderstats_flops(df, nworkers=18; onlycompute=true)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency    
    df = df[df.nworkers .== nworkers, :]
    println("worker_flops:\t$(unique(df.worker_flops))")
    println("nbytes:\t$(unique(df.nbytes))\n")
    dfo = orderstats_df(df)    

    # latency order stats recorded for individual workers
    plt.figure()
    nwait_target = nworkers
    for worker_flops in sort!(unique(dfo.worker_flops))
        dfi = dfo
        dfi = dfi[dfi.nwait .== nwait_target, :]
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        # dfi = dfi[dfi.order .<= dfi.nwait, :]       
        # plt.plot(dfi.order, dfi.worker_latency, ".", label="w_target: $nwait_target")
        straggler_fraction = sum(dfi.isstraggler) / length(dfi.isstraggler)
        dfi = dfi[dfi.isstraggler .== false, :]

        dfj = by(dfi, order_col, :worker_latency => mean => :mean, :jobid => ((x) -> length(unique(x))) => :njobs)
        sort!(dfj, order_col)
        plt.plot(dfj[order_col], dfj.mean, "o", label="w_target: $nwait_target ($worker_flops)")

        println("nwait_target: $nwait_target")
        println(collect(zip(dfj[order_col], dfj.njobs)))
        println("straggler_fraction: $straggler_fraction")
        println()

        if size(dfj, 1) >= 3
            p, coeffs = fit_polynomial(dfj[order_col], dfj.mean, 3)
            ts = range(0, nwait_target, length=100)
            # plt.plot(ts, p.(ts))
            # println(coeffs)   
        end     

        # iteration latency for different nwait_target
        dfi = df
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        # dfi = mean_latency_df(df)
        # plt.plot(dfi.nwait, dfi.latency, ".")    
        dfj = by(dfi, :nwait, :latency => mean => :mean)
        plt.plot(dfj.nwait, dfj.mean, "s", label="Iteration latency ($worker_flops)")
        
        # p, coeffs = fit_polynomial(dfj.nwait, dfj.t_compute_mean, 3)
        # ts = range(0, nworkers, length=100)
        # plt.plot(ts, p.(ts))
        # println(coeffs)    


        # latency predicted by the order statistics of Normal random variables
        meanp = Polynomial([2.3552983559702727e-17, 3.5452942951744024e-9, 6.963505495725266e-19])
        varp = Polynomial([6.3248412362377695e-22, 9.520417443453858e-14, 3.2099667366421632e-21])
        μ = meanp(worker_flops)
        σ = sqrt(varp(worker_flops))
        samples = zeros(1000)
        vs = zeros(0)
        for i in 1:nworkers
            s = OrderStatistic(Normal(μ, σ), i, nworkers)
            Distributions.rand!(s, samples)
            push!(vs, mean(samples))
        end
        plt.plot(1:nworkers, vs, "--")
    end

    plt.legend()
    plt.grid()
    plt.xlabel("Order")
    plt.ylabel("Latency [s]")
    return  
end

"""

Fix worker_flops
Plot the difference in latency due to waiting for 1 more worker
"""
function plot_orderstats_derivative(df, worker_flops; onlycompute=true)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    df = df[isapprox.(df.worker_flops, worker_flops, rtol=1e-2), :]
    dfo = orderstats_df(df)
    dfo = dfo[dfo.order .<= dfo.nwait, :]
    dfo.order = dfo.order ./ dfo.nworkers
    dfo.compute_order = dfo.compute_order ./ dfo.nworkers

    plt.figure()
    for nworkers in sort!(unique(dfo.nworkers))
        dfi = dfo
        dfi = dfi[dfi.nworkers .== nworkers, :]
        # return dfi

        dfj = by(dfi, order_col, latency_col => mean => :mean, :jobid => ((x) -> length(unique(x))) => :njobs)
        sort!(dfj, order_col)
        ys = diff(dfj.mean)
        plt.plot(dfj[order_col][1:end-1], ys, "-o", label="Nn: $nworkers")
    end
    plt.grid()
    plt.legend()
    plt.xlabel("w")
    plt.ylabel("Diff")
    return

    # dfo.npartitions = dfo.nworkers .* dfo.nsubpartitions
    df.order = df.order ./ df.nworkers

    # vs = [0.05140742522974717, 0.05067826288956093, 0.05122862096280494, 0.050645562535343865, 0.05099657254820253, 0.05138775739534186, 0.050901895214905575, 0.050316781109837165, 0.05188810059429005, 0.051392694196391135, 0.05082197380306366, 0.050276556499358506, 0.05169503754425293, 0.051862277178989835, 0.05151591182965395, 0.05201310404786811, 0.05083756996715019, 0.05134404845100367]    
    # vs ./= maximum(vs)
    # df.latency ./= vs[df.worker_index]

    dfi = df
    dfi = dfi[dfi.nworkers .== 18, :]
    dfi = dfi[dfi.nsubpartitions .== 2, :]
    # dfi.order = dfi.order ./ dfi.nworkers
    dfj = by(dfi, :order, :worker_latency => mean => :mean)
    sort!(dfj, :order)
    plt.plot(dfj.order[1:end-1], diff(dfj.mean), "o", label="(18, 2)")

    dfi = df
    dfi = dfi[dfi.nworkers .== 36, :]
    dfi = dfi[dfi.nsubpartitions .== 1, :]
    # dfi.order = dfi.order ./ dfi.nworkers    
    dfj = by(dfi, :order, :worker_latency => mean => :mean)
    # plt.plot(dfj.order, dfj.latency_mean, "o", label="(36, 1)")    
    plt.plot(dfj.order[1:end-1], diff(dfj.mean), "s", label="(36, 1)")

    plt.grid()
    plt.ylim(0)
    plt.xlim(0, 1)
    plt.xlabel("order / nworkers")
    plt.ylabel("Diff. latency. wrt. order.")
    plt.legend()
    return  
end

"""

Plot the distribution of the average worker latency
"""
function plot_worker_latency_moments(dfo; miniterations=100000, onlycompute=true, intervalsize=100)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    dfo = dfo[dfo.niterations .>= miniterations, :]
    # dfo = orderstats_df(df)
    dfo = dfo[dfo.order .<= dfo.nwait, :]
    dfo.interval = ceil.(Int, dfo.iteration ./ intervalsize)
    dfi = by(
        dfo, [:jobid, :worker_index, :interval, :worker_flops],
        latency_col => mean => :mean, 
        latency_col => var => :var,
        latency_col => minimum => :minimum,        
        )

    meanp = Polynomial([3.020008104166731e-17, 2.8011867905401972e-9, 3.5443625816981855e-18]) # for < 7e8
    # meanp = Polynomial([2.3552983559702727e-17, 3.5452942951744024e-9, 6.963505495725266e-19])
    varp = Polynomial([6.3248412362377695e-22, 9.520417443453858e-14, 3.2099667366421632e-21])

    # worker latency mean
    plt.figure()
    for worker_flops in sort!(unique(dfi.worker_flops))
        dfj = dfi
        dfj = dfj[dfj.worker_flops .== worker_flops, :]

        # empirical cdf
        xs = sort(dfj.mean)
        ys = 1 .- range(0, 1, length=length(xs))    
        plt.semilogy(xs, ys, label="c: $worker_flops")
        
        # normal distribution cdf fitted to the data
        rv = Distributions.fit(Normal, xs)
        xs = range(0.9*minimum(xs), 1.1*maximum(xs), length=100)    
        plt.semilogy(xs, 1 .- cdf.(rv, xs), "k--")

        # normal distribution predicted by the model
        # μ = meanp(worker_flops)
        # σ = sqrt(varp(worker_flops))
        # plt.plot(xs, cdf.(Normal(μ, σ), xs), "-.")

        # println(((μ, σ), params(rv)))
    end
    plt.legend()
    plt.grid()    
    plt.xlabel("Mean")

    # worker latency variance
    plt.figure()
    for worker_flops in sort!(unique(dfi.worker_flops))
        dfj = dfi
        dfj = dfj[dfj.worker_flops .== worker_flops, :]

        # empirical cdf
        xs = sort(dfj.var)
        ys = range(0, 1, length=length(xs))    
        plt.plot(xs, ys, ".", label="c: $worker_flops")

        # exponential distribution fit to the data
        # rv = Distributions.fit(Exponential, xs)
        # ts = range(0, maximum(xs), length=100)
        # plt.semilogy(ts, 1 .- cdf.(rv, ts), "--")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Variance")

    # minimum worker latency
    plt.figure()
    for worker_flops in sort!(unique(dfi.worker_flops))
        dfj = dfi
        dfj = dfj[dfj.worker_flops .== worker_flops, :]

        # empirical cdf
        xs = sort(dfj.minimum)
        ys = range(0, 1, length=length(xs))    
        plt.plot(xs, ys, label="c: $worker_flops")

        # normal distribution cdf fitted to the data
        rv = Distributions.fit(Normal, xs)
        xs = range(0.9*minimum(xs), 1.1*maximum(xs), length=100)    
        plt.plot(xs, cdf.(rv, xs), "k--")

        # exponential distribution fit to the data
        # rv = Distributions.fit(Exponential, xs)
        # ts = range(0, maximum(xs), length=100)
        # plt.semilogy(ts, 1 .- cdf.(rv, ts), "--")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Minimum")

    # minimum latency vs. mean
    plt.figure()
    for worker_flops in sort!(unique(dfi.worker_flops))
        dfj = dfi
        dfj = dfj[dfj.worker_flops .== worker_flops, :]
        plt.plot(dfj.mean, dfj.minimum, ".", label="c: $worker_flops")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Mean")    
    plt.ylabel("Minimum")

    return

    # worker latency mean vs. variance
    plt.figure()
    for worker_flops in sort!(unique(dfi.worker_flops))
        dfj = dfi
        dfj = dfj[dfj.worker_flops .== worker_flops, :]
        plt.plot(dfj.mean, dfj.var, ".", label="c: $worker_flops")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Mean")    
    plt.ylabel("Variance")

    return

    dfj = by(dfi, :worker_flops, :mean => mean => :mean, :mean => var => :var)

    dfj = dfj[dfj.worker_flops .< 7e8, :] # TODO: remove

    plt.figure()
    plt.plot(dfj.worker_flops, dfj.mean, ".")
    
    p, coeffs = fit_polynomial(dfj.worker_flops, dfj.mean, 2)
    println(coeffs)
    xs = range(0, maximum(dfj.worker_flops), length=100)
    plt.plot(xs, p.(xs))

    plt.xlabel("c")
    plt.ylabel("mean")
    plt.grid()

    plt.figure()
    plt.plot(dfj.worker_flops, dfj.var, ".")

    p, coeffs = fit_polynomial(dfj.worker_flops, dfj.var, 2)
    println(coeffs)    
    xs = range(0, maximum(dfj.worker_flops), length=100)
    plt.plot(xs, p.(xs))

    plt.xlabel("c")
    plt.ylabel("var")    
    plt.grid()

    return
end

"""

Plot average latency vs. worker index. Used to check for an unbalanced workload.
"""
function plot_latency_vs_index(dfo; nworkers=36)
    dfo = dfo[dfo.nworkers .== nworkers, :]
    dfo = dfo[dfo.nwait .== dfo.nworkers, :]
    dfo = dfo[dfo.nsubpartitions .== 40, :]

    # v = vec([0.046013058294510065 0.04661548733607283 0.04717171332346151 0.048663914553752165 0.04604952144607892 0.047401226679008895 0.0475423807538188 0.04825071847763072 0.048935551942152135 0.044736890596039657 0.04905559800587531 0.04656675616897368 0.04879067583789817 0.04809962375518636 0.04949227838810255 0.048543620826472816 0.04701325110412659 0.049071200809913604 0.04807153163181583 0.04561831089039209 0.04867938291185996 0.0467298886977427 0.04958418279574397 0.046713691418794605 0.04838288010242119 0.047936229493032986 0.04793848676944487 0.044739402462279566 0.047333384716176215 0.046625238376858454 0.04535431569172516 0.04669735977629103 0.048576057676228614 0.046901565040630845 0.04988098280145014 0.046766727535427696])
    v = [0.05386300123092638, 0.054933862699269115, 0.05368342685927259, 0.0542031652059991, 0.05333776666867998, 0.053177820727389205, 0.05263823982264996, 0.05321165238786512, 0.05398119330531424, 0.05366335096970345, 0.05374068401736961, 0.05400672957786409, 0.05330386859234326, 0.05284493331388487, 0.05410580473945859, 0.0529785826439725, 0.05449267995850014, 0.053815433081681414, 0.053770737794055855, 0.05322785079874978, 0.05389343286675174, 0.054128245609753725, 0.05393979259819193, 0.05313907654430421, 0.05383527758559796, 0.05353642567903311, 0.0537138425740668, 0.05215185214435854, 0.05347711205441357, 0.05485612366530882, 0.05355829799161612, 0.053100017485247016, 0.05346381763960923, 0.05314798347672649, 0.05323480333614919, 0.053582819563781185]

    plt.figure()
    for worker_flops in sort!(unique(dfo.worker_flops))
        dfi = dfo
        dfi = dfi[dfi.worker_flops .== worker_flops, :]
        dfi = by(dfi, :worker_index, :worker_latency => mean => :worker_latency)
        sort!(dfi, :worker_index)
        plt.plot(v ./ mean(v), dfi.worker_latency ./ mean(dfi.worker_latency), "o", label="c: $worker_flops")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("worker index")
    plt.ylabel("Avg. latency")
    return
end

"""

Compute Markov process state transition probability matrix
"""
function compute_markov_state_matrix(df)
    intervals = [100, 10, 0.1, 0.01]

    vs = df["rmean_10.0"]
    states = sort!(unique(round.(vs, digits=4)))
    state = ceil.(Int, (vs .- states[1]) / (states[end] - states[1]) * length(states))


    P = zeros(length(states)+1, length(states)+1)
    for i in 2:length(state)
        P[state[i-1], state[i]] += 1
    end
    for i in 1:size(P, 1)
        P[i, :] ./= sum(P[i, :])
    end
    return P
    # C = zeros(Int, length(states), length(states))

    plt.figure()    
    plt.plot(df.time, vs)

    # for interval in intervals
    #     plt.plot(df.time, df["rmean_$interval"], ".", label="$interval")
    # end            

    plt.legend()
    plt.grid()
    return
end

function variance_test(dfo)

    shift = 1.04452512147574e-9
    # dfo = dfo[dfo.burst .== false, :]
    # dfo.straggling = dfo.worker_compute_latency ./ dfo.worker_flops .- shift

    # dfo = by(dfo, [:jobid, :worker_index, :worker_flops], :straggling => mean => :mean)
    # return dfo

    plt.figure()
    plt.plot(dfo.worker_flops, dfo.mean, ".")
    plt.grid()

    return
    df = by(dfo, :worker_flops, 
        :worker_compute_latency => minimum => :minimum,
        :worker_compute_latency => ((x) -> quantile(x, 0.01)) => :q001,
        :worker_compute_latency => ((x) -> quantile(x, 0.01)) => :q002,
        :worker_compute_latency => ((x) -> quantile(x, 0.01)) => :q005,
        :worker_compute_latency => ((x) -> quantile(x, 0.01)) => :q01,
        )
    
    plt.figure()            

    ys = (df.minimum .- df.worker_flops.*shift) ./ df.worker_flops
    plt.plot(df.worker_flops, ys, "o", label="Minimum")

    ys = (df.q001 .- df.worker_flops.*shift) ./ df.worker_flops
    plt.plot(df.worker_flops, ys, "o", label="q001")    

    ys = (df.q002 .- df.worker_flops.*shift) ./ df.worker_flops
    plt.plot(df.worker_flops, ys, "o", label="q002")    
    
    ys = (df.q005 .- df.worker_flops.*shift) ./ df.worker_flops
    plt.plot(df.worker_flops, ys, "o", label="q005")        

    ys = (df.q01 .- df.worker_flops.*shift) ./ df.worker_flops
    plt.plot(df.worker_flops, ys, "o", label="q01")            


    plt.legend()
    plt.grid()
    return df
end

function markov_test()

    nstates = 100
    P = zeros(nstates, nstates)
    for i in 1:size(P, 1)
        P[i, i] = 0.9
    end
    for i in 2:size(P, 1)-1
        P[i-1, i] = 0.05
        P[i+1, i] = 0.05
    end    
    P[2, 1] = 0.1
    P[end-1, end] = 0.1
    x = zeros(nstates)
    x[round(Int, nstates/2)] = 1

    vars = zeros(1000)
    for i in 1:1000
        x .= P*x
        vars[i] = var(x)
    end
    plt.figure()
    plt.plot(vars)
    return

    plt.figure()
    for _ in 1:10
        x .= P*x
    end
    plt.plot(x, label="10")
    println(var(x))

    for _ in 1:90
        x .= P*x
    end
    plt.plot(x, label="100")    
    println(var(x))

    for _ in 1:900
        x .= P*x
    end
    plt.plot(x, label="1000")        
    println(var(x))
end

function exponential_test()

    rv = Exponential()
    # scale_rv = Exponential()
    nsamples = 10000

    plt.figure()
    means = zeros(0)
    vars = zeros(0)
    cs = [10, 20, 30, 100, 200, 500, 1000]
    for c in cs
        scale_rv = Exponential(sqrt(c))

        shifts = ones(nsamples, c)
        scales = rand(scale_rv, nsamples, c)

        latency = vec(sum(shifts, dims=2))
        latency .+= vec(sum(rand(rv, nsamples, c) .* scales, dims=2))
        # latency = vec(sum(shifts .+ rand(rv, nsamples, c) .* scales, dims=2))
        # latency ./= c
        sort!(latency)
        ys = range(0, 1, length=nsamples)
        plt.plot(latency ./ c, ys, label="c: $c")
        println("[c: $c] mean: $(mean(latency))")

        push!(means, mean(latency))
        push!(vars, var(latency))
    end
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(cs, means, ".")
    plt.legend()
    plt.ylabel("Mean")    
    plt.grid()

    plt.figure()
    plt.plot(cs, vars, ".")
    plt.ylabel("Variance")
    plt.grid()

    return

    nvariables = 10000
    nsamples = 1000
    samples = zeros(nsamples)
    rv = Exponential()
    for i in 1:nsamples
        β = 1.0
        v = 0.0
        for _ in 1:nvariables
            v += rand(rv) * β
            if rand() < 0.5
                β *= 0.95
            else
                β *= 1.05
            end
        end
        samples[i] = v
    end
    sort!(samples)
    ys = range(0, 1, length=nsamples)
    plt.figure()
    plt.plot(samples, ys)

    rv = Distributions.fit(Normal, samples)
    ts = range(quantile(rv, 1e-6), quantile(rv, 1.0-1e-6), length=100)
    plt.plot(ts, cdf.(rv, ts), "k--")

    rv = Distributions.fit(Gamma, samples)
    ts = range(quantile(rv, 1e-6), quantile(rv, 1.0-1e-6), length=100)
    plt.plot(ts, cdf.(rv, ts), "r--")

    plt.title("$((nvariables))")
    plt.grid()
    return
end

function sparse_dense_test()
    # let's look at how latency scales with the number of rows of the matrix for dense and sparse matrices
    nsamples = 10
    ncols = 1000
    ncomponents = 3
    nrows_all = [1000, 2000, 3000, 4000, 5000, 10000, 20000, 50000, 70000, 100000]
    latency_sparse = zeros(length(nrows_all))
    latency_dense = zeros(length(nrows_all))
    for (i, nrows) in enumerate(nrows_all)
        # X_dense = randn(nrows, ncols)
        X_sparse = sprand(nrows, ncols, 0.05)
        V = zeros(ncols, ncomponents)
        W = zeros(nrows, ncomponents)
        # for _ in 1:nsamples
        #     latency_dense[i] += @elapsed mul!(W, X_dense, V)
        #     latency_dense[i] += @elapsed mul!(V, X_dense', W)
        # end
        # latency_dense[i] /= nsamples

        for _ in 1:nsamples
            latency_sparse[i] += @elapsed mul!(W, X_sparse, V)
            latency_sparse[i] += @elapsed mul!(V, X_sparse', W)
        end
        latency_sparse[i] /= nsamples        
    end

    flops = 2 .* nrows_all .* ncols .* ncomponents

    plt.figure()
    # plt.plot(flops, latency_dense, "s")
    plt.plot(flops, latency_sparse, "o")
    plt.grid()
    return
end

function plot_iteration_distribution(dfo)

    # all iterations
    plt.figure()
    for worker_flops in sort!(unique(dfo.worker_flops))
        dfi = dfo
        xs = dfi[dfi.worker_flops .== worker_flops, :worker_compute_latency]
        sort!(xs)
        ys = range(0, 1, length=length(xs))
        plt.semilogy(xs, 1.0.-ys, label="c: $worker_flops")
    end
    plt.legend()
    plt.grid()
    
    # only the first iteration
    dfo = dfo[dfo.iteration .== 1, :]
    plt.figure()    
    for worker_flops in sort!(unique(dfo.worker_flops))
        dfi = dfo
        xs = dfi[dfi.worker_flops .== worker_flops, :worker_compute_latency]
        sort!(xs)
        ys = range(0, 1, length=length(xs))
        plt.semilogy(xs, 1.0.-ys, label="c: $worker_flops")
    end
    plt.legend()
    plt.grid()    
end

function tail_plot(dfo)

    dfo = copy(dfo)    
    dfo.slot = ceil.(Int, dfo.time ./ 10) # split into slots
    df = by(
        dfo, [:jobid, :worker_index, :slot, :worker_flops],
        :worker_compute_latency => mean => :mean, 
        :worker_compute_latency => var => :var, 
        :worker_compute_latency => median => :median,
        :worker_compute_latency => minimum => :minimum,
        )

    # dfo = dfo[dfo.burst .== false, :]    
    # intersect = 2.5461674786469558e-17
    # slope = 8.121166381604497e-10    
    # dfo.noise = dfo.worker_compute_latency .- slope .* dfo.worker_flops
    # df = by(dfo, [:jobid, :worker_index, :worker_flops], :noise => mean => :mean, :noise => var => :var, :noise => median => :median)

    # mean and median
    plt.figure()
    plt.plot(df.worker_flops, df.mean, "o", label="Mean")
    plt.plot(df.worker_flops, df.minimum, ".", label="Minimum")
    # plt.plot(df.worker_flops, df.median, ".", label="Median")
    plt.ylabel("Latency [s]")
    plt.xlabel("Flops")
    plt.legend()
    plt.grid()

    # variance
    plt.figure()
    plt.plot(df.worker_flops, df.var, ".")
    plt.ylabel("Latency variance")
    plt.xlabel("Flops")
    plt.grid()

    return
end

"""

"""
function plot_worker_latency_timeseries(dfo; worker_flops)
    dfo = dfo[isapprox.(dfo.worker_flops, worker_flops, rtol=1e-2), :]    
    # select a job at random
    jobid = rand(unique(dfo.jobid))
    dfo = dfo[dfo.jobid .== jobid, :]
    dfo = dfo[dfo.iteration .<= 1000, :]
    plt.figure()    

    worker_index = rand(1:unique(dfo.nworkers)[1])
    dfi = dfo[dfo.worker_index .== worker_index, :]
    sort!(dfi, :iteration)
    comm_latency = dfi.worker_latency .- dfi.worker_compute_latency
    plt.plot(dfi.iteration, dfi.worker_compute_latency, "b-", label="Worker 1 (compute)")
    plt.plot(dfi.iteration, comm_latency, "r-", label="Worker 1 (communication)")
    write_table(dfi.iteration, dfi.worker_compute_latency, "comp_latency_1.csv")
    write_table(dfi.iteration, comm_latency, "comm_latency_1.csv")

    worker_index = rand(1:unique(dfo.nworkers)[1])
    dfi = dfo[dfo.worker_index .== worker_index, :]
    sort!(dfi, :iteration)
    comm_latency = dfi.worker_latency .- dfi.worker_compute_latency
    plt.plot(dfi.iteration, dfi.worker_compute_latency, "c-", label="Worker 2 (compute)")
    plt.plot(dfi.iteration, comm_latency, "m-", label="Worker 2 (communication)")    
    write_table(dfi.iteration, dfi.worker_compute_latency, "comp_latency_2.csv")
    write_table(dfi.iteration, comm_latency, "comm_latency_2.csv")

    plt.legend()
    plt.grid()
    plt.xlabel("Latency [s]")
    plt.ylabel("Iteration")
end

"""

Plot worker latency over time
"""
function plot_worker_latency(df, n=10; miniterations=10000, onlycompute=true, worker_flops)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    df = df[df.niterations .>= miniterations, :]
    df = df[isapprox.(df.worker_flops, worker_flops, rtol=1e-2), :]

    # select the job with the highest recorded latency
    # i = argmin(df.worker_compute_latency)
    # is = sortperm(df.worker_compute_latency)
    # i = is[end-10]
    # jobid = df.jobid[i]
    # worker_index = df.worker_index[i]

    # select a job and worker at random
    jobid = rand(unique(df.jobid))
    worker_index = rand(1:36)
    # jobid, worker_index = 637, 32

    println("jobid: $jobid, worker_index: $worker_index")
    df = df[df.jobid .== jobid, :]
    df = df[df.worker_index .== worker_index, :]

    # compute running mean over windows of varying size
    windowlengths = [Inf, 5, 0]
    # df = compute_rmeans(df; windowlengths)

    # plot timeseries latency
    plt.figure()    
    plt.plot(df.time, df.worker_compute_latency, ".", label="Total latency")
    # dfi = df[df.burst, :]    
    # plt.plot(dfi.time, dfi.worker_compute_latency, "o", label="Burst latency")
    # for windowlength in reverse(windowlengths[2:end])
    #     plt.plot(df.time, df["rmean_$windowlength"], ".", label="$windowlength")
    # end

    # plt.plot(df.time, df["rmean_0.0"], ".", label="Latency - (Mean + Markov)") 
    # plt.plot(df.time, df["rmean_10.0"], "-", label="Markov process")

    plt.grid()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Latency [s]")
    return

    ### plot latency distribution
    plt.figure()

    # empirical
    xs = sort(df.worker_compute_latency)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys)

    # fitted
    rv = Distributions.fit(Gamma, xs)
    ts = range(quantile(rv, 1e-6), quantile(rv, 1.0-1e-6), length=100)
    plt.plot(ts, cdf.(rv, ts), "k--")

    plt.grid()
    plt.xlabel("Latency [s]")
    plt.ylabel("CDF")
    return
end

function plot_latency_noise(dfo; miniterations=10000, onlycompute=true)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    dfo = dfo[dfo.niterations .>= miniterations, :]

end

"""

Fit all random variables determining the latency of a worker
"""
function fit_worker_latency_process(dfo, worker_flops; miniterations=10000, onlycompute=true)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    dfo = dfo[dfo.niterations .>= miniterations, :]
    dfo = dfo[isapprox.(dfo.worker_flops, worker_flops, rtol=1e-2), :]
    dfo = dfo[dfo.nwait .== dfo.nworkers, :] # ensure all workers are available at the start of each iteration
    dfo.burst = burst_state_from_orderstats_df(dfo)

    # latency outside of and during bursts
    df1 = dfo[dfo.burst .== false, :]
    df2 = dfo[dfo.burst, :]

    # distribution of the mean latency of each worker, outside of bursts
    dfi = by(df1, [:jobid, :worker_index], latency_col => mean => :mean)    
    rv_shift = Distributions.fit(Normal, dfi.mean)

    # add the mean latency outside of bursts to df1 and df2
    df1 = leftjoin(df1, dfi, on=[:jobid, :worker_index])
    df2 = leftjoin(df2, dfi, on=[:jobid, :worker_index])

    # distribution of the latency noise outside of bursts
    df1[latency_col] .-= df1.mean
    dfj = by(df1, [:jobid, :worker_index], latency_col => mean => :mean, latency_col => var => :var)
    rv_mean = Distributions.fit(Normal, filter(!isnan, dfj.mean))
    rv_var = Distributions.fit(LogNormal, filter(!isnan, dfj.var))

    # distribution of the latency noise during bursts
    df2[latency_col] .-= df2.mean
    dfj = by(df2, [:jobid, :worker_index], latency_col => mean => :mean, latency_col => var => :var)
    if size(dfj, 1) > 0
        rv_mean_burst = Distributions.fit(Normal, filter(!isnan, dfj.mean))
        rv_var_burst = Distributions.fit(LogNormal, filter(!isnan, dfj.var))
    else
        rv_mean_burst = nothing
        rv_var_burst = nothing
    end

    # burst state transition matrix
    P = zeros(2, 2)
    for i in 1:(size(dfo, 1)-1)
        current_state = dfo.burst[i] ? 2 : 1
        next_state = dfo.burst[i+1] ? 2 : 1
        P[current_state, next_state] += 1
    end
    for i in 1:size(P, 1)
        P[i, :] ./= sum(P[i, :])
    end

    return rv_shift, rv_mean, rv_var, rv_mean_burst, rv_var_burst, P
end

"""

Latency process model fitted to the data
"""
function models_from_df(dfo; miniterations=10000, onlycompute=true)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    dfo = dfo[dfo.niterations .>= miniterations, :]
    dfo = dfo[dfo.nwait .== dfo.nworkers, :] # ensure all workers are available at the start of each iteration
    worker_flops = sort!(unique(dfo.worker_flops))
    models = [fit_worker_latency_process(dfo, c) for c in worker_flops]
    worker_flops, models
end

"""

Fit all random variables determining the latency of a worker
"""
function plot_worker_latency_process(worker_flops, models)

    ### avg. latency outside of bursts
    # latency distribution
    # plt.figure()
    # println("Latency shift")
    # for (c, model) in zip(worker_flops, models)
    #     rv_shift, rv_mean, rv_var, rv_mean_burst, rv_var_burst, P = model
    #     if !isnothing(rv_shift)
    #         rv = rv_shift
    #         println(rv)
    #         ts = range(quantile(rv, 1e-6), quantile(rv, 1.0-1e-6), length=100)
    #         plt.semilogy(ts, 1 .- cdf.(rv, ts), label="flops: $(round(c, sigdigits=3))")
    #     end        
    # end
    # plt.legend()
    # plt.grid()
    # # plt.title("Avg. latency distribution")
    # plt.xlabel("Time [s]")
    # plt.ylabel("CCDF")

    # # plot the mean vs. c
    # plt.figure()
    # ys = [params(model[1])[1] for model in models]
    # plt.plot(worker_flops, ys, "-o", label="Empirical")
    # plt.grid()
    # plt.ylabel("Avg. latency μ")
    # plt.xlabel("flops")

    # p, coeffs = fit_polynomial(worker_flops, ys, 2)
    # println("Avg. latency μ: $coeffs")
    # ts = range(minimum(worker_flops), maximum(worker_flops), length=100)
    # plt.plot(ts, p.(ts), "k--", label="Fitted")
    # plt.legend()

    # # plot the variance vs. c
    # plt.figure()
    # ys = [params(model[1])[2] for model in models]
    # plt.plot(worker_flops, ys, "-o", label="Empirical")
    # plt.grid()    
    # plt.ylabel("Avg. latency σ")
    # plt.xlabel("flops")

    # p, coeffs = fit_polynomial(worker_flops, ys, 2)
    # println("Avg. latency σ: $coeffs")
    # ts = range(minimum(worker_flops), maximum(worker_flops), length=100)
    # plt.plot(ts, p.(ts), "k--", label="Fitted")    
    # plt.legend()

    ### latency noise outside of bursts
    # plot the distribution of μ_A
    # plt.figure()
    # println("Latency noise μ")    
    # for (c, model) in zip(worker_flops, models)
    #     rv_shift, rv_mean, rv_var, rv_mean_burst, rv_var_burst, P = model
    #     if !isnothing(rv_mean)
    #         rv = rv_mean
    #         println(rv)
    #         ts = range(quantile(rv, 1e-6), quantile(rv, 1.0-1e-6), length=100)
    #         plt.plot(ts, cdf.(rv, ts), label="c: $(round(c, sigdigits=3))")
    #     end
    # end
    # plt.legend()
    # plt.grid()
    # plt.xlabel("\$\\mu_A\$")
    # plt.ylabel("CDF")

    # # plot the distribution of σ_A
    # plt.figure()
    # println("Latency noise σ")    
    # for (c, model) in zip(worker_flops, models)
    #     rv_shift, rv_mean, rv_var, rv_mean_burst, rv_var_burst, P = model
    #     if !isnothing(rv_mean)
    #         rv = rv_var
    #         println(rv)
    #         ts = range(quantile(rv, 1e-6), quantile(rv, 1.0-1e-6), length=100)
    #         plt.plot(ts, cdf.(rv, ts), label="c: $(round(c, sigdigits=3))")
    #     end
    # end
    # plt.legend()
    # plt.grid()
    # plt.xlabel("\$\\sigma_A\$")
    # plt.ylabel("CDF")

    # plot mean(σ_A)
    plt.figure()
    # ys = [params(model[3])[1] for model in models]
    ys = [mean(model[3]) for model in models]

    xs = worker_flops
    # xs = log.(worker_flops)
    # xs = log.([mean(model[1]) for model in models])
    # xs = [var(model[1]) for model in models]

    plt.plot(xs, ys, "o", label="Empirical")
    plt.grid()
    plt.ylabel("mean(\$\\sigma_A\$)")
    
    plt.xlabel("c")
    # plt.xlabel("log(c)")
    # plt.xlabel("μ_S")
    # plt.xlabel("σ_S")

    # p, coeffs = fit_polynomial(worker_flops, ys, 2)
    # println("Latency noise variance μ: $coeffs")
    # ts = range(minimum(worker_flops), maximum(worker_flops), length=100)
    # plt.plot(ts, p.(ts), "k--", label="Fitted")

    plt.legend()

    # plot var(σ_A)
    plt.figure()
    # ys = [params(model[3])[2] for model in models]
    ys = [var(model[3]) for model in models]

    xs = worker_flops
    # xs = log.(worker_flops)
    # xs = log.([mean(model[1]) for model in models])
    # xs = [var(model[1]) for model in models]

    plt.plot(xs, ys, "o", label="Empirical")
    plt.grid()
    plt.ylabel("var(\$\\sigma_A\$)")
    
    plt.xlabel("c")
    # plt.xlabel("log(c)")
    # plt.xlabel("μ_S")    
    # plt.xlabel("σ_S")

    # p, coeffs = fit_polynomial(worker_flops, ys, 2)
    # println("Latency noise variance σ: $coeffs")
    # ts = range(minimum(worker_flops), maximum(worker_flops), length=100)
    # plt.plot(ts, p.(ts), "k--", label="Fitted")
    
    plt.legend()        

    return

    ### latency noise during bursts
    plt.figure()
    println("Latency noise μ (burst)")
    for (c, model) in zip(worker_flops, models)
        rv_shift, rv_mean, rv_var, rv_mean_burst, rv_var_burst, P = model
        if !isnothing(rv_mean_burst)
            rv = rv_mean_burst
            println(rv)            
            ts = range(quantile(rv, 1e-2), quantile(rv, 1.0-1e-2), length=100)            
            plt.plot(ts, cdf.(rv, ts), "--", label="c: $c (burst)")
        end
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Latency noise μ (burst)")    

    plt.figure()
    println("Latency noise σ (burst)")
    for (c, model) in zip(worker_flops, models)
        rv_shift, rv_mean, rv_var, rv_mean_burst, rv_var_burst, P = model
        if !isnothing(rv_mean_burst)
            rv = rv_var_burst
            println(rv)            
            ts = range(quantile(rv, 1e-2), quantile(rv, 1.0-1e-2), length=100)            
            plt.plot(ts, cdf.(rv, ts), "--", label="c: $c (burst)")
        end
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Latency noise σ (burst)")

    return
end

function plot_latency_vs_nflops(dfo; onlycompute=false)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    dfo = dfo[dfo.nwait .== dfo.nworkers, :] # ensure all workers are available at the start of each iteration
    df = by(dfo, :worker_flops, latency_col => mean => :mean)
    sort!(df, :worker_flops)
    # plt.figure()
    plt.plot(dfo.worker_flops, dfo.worker_latency, ".")
    plt.plot(df.worker_flops, df.mean, "-o")
    plt.grid()
    return df
end

"""

Compute the state transition matrix associated with latency bursts.
"""
function compute_burst_state_matrix(dfo; miniterations=10000, worker_flops=nothing)
    dfo = dfo[dfo.niterations .>= miniterations, :]
    if !isnothing(worker_flops)
        dfo = dfo[isapprox.(dfo.worker_flops, worker_flops, rtol=1e-2), :]
    end
    sort!(dfo, [:jobid, :worker_index, :iteration])
    dfo.burst = burst_state_from_orderstats_df(dfo)

    state = 1 .+ dfo.burst    
    P = zeros(2, 2)
    for i in 2:size(dfo, 1)
        P[state[i-1], state[i]] += 1
    end
    P[1, :] ./= sum(P[1, :])
    P[2, :] ./= sum(P[2, :])
    P
end


"""

Plot the latency distribution during bursts.
"""
function plot_bursts(dfo; miniterations=10000, worker_flops=nothing)
    dfo = dfo[dfo.niterations .>= miniterations, :]
    if !isnothing(worker_flops)
        dfo = dfo[isapprox.(dfo.worker_flops, worker_flops, rtol=1e-2), :]
    end
    sort!(dfo, [:jobid, :worker_index, :iteration])
    dfo.burst = burst_state_from_orderstats_df(dfo)

    # plot latency outside and during bursts seprately
    # plt.figure()
    # n = 5
    # for _ in 1:n

    #     # select a job and worker at random        
    #     dfi = dfo
    #     jobid = rand(unique(dfi.jobid))
    #     worker_index = rand(1:36)
    #     println("jobid: $jobid, worker_index: $worker_index")
    #     dfi = dfi[dfi.jobid .== jobid, :]
    #     dfi = dfi[dfi.worker_index .== worker_index, :]

    #     dfj = dfi[dfi.burst, :]
    #     plt.plot(dfj.time, dfj.worker_compute_latency, "^")

    #     dfj = dfi[dfi.burst .== false, :]
    #     plt.plot(dfj.time, dfj.worker_compute_latency, ".")        
    # end    
    # plt.grid()
    # return

    # mean and variance of the latency during bursts
    dfi = dfo[dfo.burst, :]
    df1 = by(
        dfi, [:jobid, :worker_index], 
        :worker_compute_latency => mean => :mean,
        :worker_compute_latency => var => :var,
        )

    # subtract the mean computed outside bursts
    dfi = dfo[dfo.burst .== false, :]
    df2 = by(
        dfi, [:jobid, :worker_index], 
        :worker_compute_latency => mean => :shift,
    )
    dfj = innerjoin(df1, df2, on=[:jobid, :worker_index])
    dfj.mean .-= dfj.shift

    # plot mean
    plt.figure()
    xs = sort(dfj.mean)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys)


    rv = Distributions.fit(Normal, view(xs, ceil(Int, 0.03*length(xs)):length(xs)))
    ts = range(0, 1.1*xs[end], length=100)
    plt.plot(ts, cdf.(rv, ts), "k--")    
    println("Burst mean RV: $rv")

    plt.grid()
    plt.xlabel("Mean")    

    # plot variance
    plt.figure()
    xs = sort(dfj.var)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys)

    rv = Distributions.fit(LogNormal, view(xs, ceil(Int, 0.03*length(xs)):length(xs)))    
    ts = range(0, 1.1*xs[end], length=100)
    plt.plot(ts, cdf.(rv, ts), "k--")
    println("Burst variance RV: $rv")

    plt.grid()
    plt.xlabel("Variance")    
    return dfj

    
    # collect the differences in latency between the mean outside of bursts and during bursts
    # over all jobs and workers
    plt.figure()
    xs = zeros(0)    
    for jobid in unique(dfo.jobid)
        dfi = dfo
        dfi = dfi[dfi.jobid .== jobid, :]
        for worker_index in unique(dfi.worker_index)
            dfj = dfi
            dfj = dfj[dfj.worker_index .== worker_index, :]
            shift = mean(dfi[dfi.burst .== false, :worker_compute_latency])
            append!(xs, dfi[dfi.burst .== true, :worker_compute_latency] .- shift)
        end
    end

    # plot the cdf
    sort!(xs)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys)

    # Normal fitted to the data
    rv = Distributions.fit(Normal, xs)
    ts = range(1.1xs[1], 1.1xs[end], length=100)
    plt.plot(ts, cdf.(rv, ts), "k--")
    println("Burst latency RV: $rv")

    plt.grid()
    return

    # plot the latency distribution during bursts for individual workers
    plt.figure()
    n = 10
    for _ in 1:n

        # select a job and worker at random        
        dfi = dfo
        jobid = rand(unique(dfi.jobid))
        worker_index = rand(1:36)
        #jobid, worker_index = 618, 3
        println("jobid: $jobid, worker_index: $worker_index")
        dfi = dfi[dfi.jobid .== jobid, :]
        dfi = dfi[dfi.worker_index .== worker_index, :]

        # mean latency sans bursts
        shift = mean(dfi[dfi.burst .== false, :worker_compute_latency])

        # additional latency during bursts
        xs = sort(dfi[dfi.burst .== true, :worker_compute_latency]) .- shift
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys)
    end

    # plt.plot(dfo.time, dfo.worker_compute_latency, ".")
    plt.grid()
    return

end

function plot_worker_latency_qq(dfo, n=10; miniterations=10000, onlycompute=true, worker_flops)
    order_col = onlycompute ? :compute_order : :order
    latency_col = onlycompute ? :worker_compute_latency : :worker_latency
    dfo = dfo[dfo.niterations .>= miniterations, :]
    dfo = dfo[isapprox.(dfo.worker_flops, worker_flops, rtol=1e-2), :]
    dfo.burst = burst_state_from_orderstats_df(dfo)
    dfo = dfo[dfo.burst .== false, :]    
    dfo = by(
        dfo, [:jobid, :worker_index, :worker_flops],
        latency_col => diff => :diff,        
        # latency_col => mean => :mean, 
        # latency_col => median => :median,
        # latency_col => var => :var,
        # latency_col => minimum => :minimum,        
        )

    # Tukey-Lambda Q-Q plot
    plt.figure()
    dfi = dfo
    xs = sort!(dfi.diff)
    ys = range(0, 1, length=length(xs))

    for λ in [-0.4]
        rv = TukeyLambda(λ)

        qs = range(0.05, 0.95, length=100)
        xs = [quantile(xs, q) for q in qs]
        scale = maximum(abs.(xs))
        xs ./= maximum(abs.(xs))
        ys = [quantile(rv, q) for q in qs]
        ys ./= maximum(abs.(ys))
        plt.plot(xs, ys, label="λ: $λ")        

        xs = quantile.(rv, [0.01, 0.99])
        scale /= maximum(abs.(xs))
        xs ./= maximum(abs.(xs))
        ys = quantile.(rv, [0.01, 0.99])
        ys ./= maximum(abs.(ys))
        plt.plot(xs, ys, "k-")                    

        println("λ: $λ, scale: $scale")
    end

    plt.xlabel("Data")
    plt.ylabel("Theoretical distribution")
    # plt.axis("equal")
    plt.grid()
    plt.legend()
    return

    plt.figure()
    for _ in 1:n
        dfi = dfo
        jobid = rand(unique(dfi.jobid))
        dfi = dfi[dfi.jobid .== jobid, :]
        worker_index = rand(unique(dfi.worker_index))
        dfi = dfi[dfi.worker_index .== worker_index, :]
        plt.plot(diff(dfi.mean), ".", label="job: $jobid, worker: $worker_index")
    end
    plt.legend()
    plt.grid()
    plt.xlabel("Iteration index")
    plt.ylabel("Latency")
end