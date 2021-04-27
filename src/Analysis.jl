using CSV, DataFrames, PyPlot, Statistics, Polynomials, LinearAlgebra, Distributions, RollingFunctions
using StatsBase

"""

Fit a linear model (i.e., a line) to the data X, y.
"""
function linear_model(X::AbstractMatrix, y::AbstractVector)
    size(X, 1) == length(y) || throw(DimensionMismatch("X has dimensions $(size(X)), but y has dimension $(length(y))"))
    A = ones(size(X, 1), size(X, 2)+1)    
    A[:, 2:end] .= X
    A \ y
end

"""

Fit a polynomial of degree `d` to the data `xs` and `ys`
"""
function fit_polynomial(xs::AbstractVector, ys::AbstractVector, d=1)
    A = ones(length(xs), d+1)
    for i in 1:d
        A[:, i+1] .= xs.^i
    end
    try
        coeffs = A\ys
        return (x) -> dot(coeffs, [x^i for i in 0:d]), coeffs        
    catch SingularException
        return (x) -> 0, fill(NaN, d+1)
    end
end

linear_model(x::AbstractVector, y::AbstractVector) = linear_model(reshape(x, length(x), 1), y)

"""

Write xs and ys as a table with columns separated by a space
"""
function write_table(xs::AbstractVector, ys::AbstractVector, filename::AbstractString)
    length(xs) == length(ys) || throw(DimensionMismatch("xs has dimension $(length(xs)), but ys has dimension $(length(ys))"))
    open(filename, "w") do io
        for i in 1:length(xs)
            write(io, "$(xs[i]) $(ys[i])\n")
        end
    end
    return
end

"""

Extend a tall DataFrame, such that there is order statistics latency for numbers of workers other than `nwait`.
"""
function extend_dfo(dfo)
    rv = DataFrame()
    dfo = filter([:nwait, :nworkers] => (x, y)->(x == y), dfo)
    dfo.mse = missing
    for i in 1:maximum(df.nworkers)
        dfi = dfo
        dfi = filter([:nworkers, :worker_index] => (x, y)->((x >= i) && (y <= i)), dfi)
        if size(dfi, 1) == 0
            continue
        end
        dfi.nworkers .= i
        dfi.nwait .= i
        sort!(dfi, [:jobid, :iteration, :worker_latency])        
        dfi.order .= by(dfi, [:jobid, :iteration], :nworkers => ((x) -> collect(1:maximum(x))) => :order).order
        append!(rv, dfi, cols=:union)
        GC.gc()
    end
    rv
end

"""

Update the worker indices such that the fastest worker has index 1 and so on.
"""
function reindex_workers_by_order!(df)
    maxworkers = maximum(df.nworkers)
    latencies = zeros(maxworkers)
    repochs = zeros(Int, maxworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    repoch_columns = ["repoch_worker_$i" for i in 1:maxworkers]
    for i in 1:size(df, 1)
        latencies .= Inf
        nworkers = df.nworkers[i]
        for j in 1:nworkers
            repoch = df[i, repoch_columns[j]]
            isstraggler = ismissing(repoch) || repoch < df.iteration[i]
            latencies[j] = isstraggler ? Inf : df[i, latency_columns[j]]
        end
        latencies_view = view(latencies, 1:nworkers)
        repochs_view = view(repochs, 1:nworkers)
        p = sortperm(latencies_view)
        for j in 1:nworkers
            df[i, latency_columns[j]] = latencies[p[j]]
            df[i, repoch_columns[j]] = repochs[p[j]]
        end
    end
    df
end

"""

Return a df composed of the order statistic samples for each worker, iteration, and job.
"""
function tall_from_wide(df)
    df = df[.!ismissing.(df["latency_worker_1"]), :]    
    if size(df, 1) == 0
        return DataFrame()
    end    

    # determine the number of workers based on column names
    # (since not all rows may be present)
    cols = [name for name in names(df) if occursin("repoch", name)]
    indices = parse.(Int, last.(split.(cols, "_")))
    nworkers = maximum(indices)
    
    ### stack by worker latency
    df1 = select(df, Not(["repoch_worker_$i" for i in 1:nworkers]))
    if "compute_latency_worker_1" in names(df)
        select!(df1, Not(["compute_latency_worker_$i" for i in 1:nworkers]))    
    end    
    df1 = stack(df1, ["latency_worker_$i" for i in 1:nworkers], value_name=:worker_latency)
    select!(df1, Not(:variable), :variable => ((x) -> parse.(Int, last.(split.(x, "_")))) => :worker_index)
    dropmissing!(df1, :worker_latency)
    GC.gc()
    
    # stack by worker receive epoch
    df2 = stack(df, ["repoch_worker_$i" for i in 1:nworkers], [:jobid, :iteration], value_name=:repoch)
    select!(df2, Not(:variable), :variable => ((x) -> parse.(Int, last.(split.(x, "_")))) => :worker_index)    
    dropmissing!(df2, :repoch)
    df2[:isstraggler] = df2.repoch .< df2.iteration
    joined = innerjoin(df1, df2, on=[:jobid, :iteration, :worker_index]) 
    GC.gc()    

    # stack by worker compute latency
    if "compute_latency_worker_1" in names(df)
        df3 = stack(df, ["compute_latency_worker_$i" for i in 1:nworkers], [:jobid, :iteration], value_name=:worker_compute_latency)
        select!(df3, Not(:variable), :variable => ((x) -> parse.(Int, last.(split.(x, "_")))) => :worker_index)            
        dropmissing!(df3, :worker_compute_latency)
        joined = innerjoin(joined, df3, on=[:jobid, :iteration, :worker_index])
    end
    GC.gc()    

    # the latency of stragglers is infinite
    joined[joined.isstraggler, :worker_latency] .= Inf

    # compute the order of all workers
    sort!(joined, [:jobid, :iteration, :worker_latency])
    joined[:order] = by(joined, [:jobid, :iteration], :nworkers => ((x) -> collect(1:maximum(x))) => :order).order
    if "compute_latency_worker_1" in names(df)
        sort!(joined, [:jobid, :iteration, :worker_compute_latency])
        joined[:compute_order] = by(joined, [:jobid, :iteration], :nworkers => ((x) -> collect(1:maximum(x))) => :order).order        
    end
    GC.gc()

    return joined
end

"""

Remove columns not necessary for analysis (to save space).
"""
function strip_columns!(df)
    for column in ["iteratedataset", "saveiterates", "outputdataset", "inputdataset", "inputfile", "algorithm", "outputfile"]
        select!(df, Not(column))
    end
    df
end

"""

For each job, to remove initialization delay (e.g., due to compilation), replace the latency of 
the first iteration with the average latency of all subsequent iteration for that job.
"""
function remove_initialization_latency!(df)
    for jobid in unique(df.jobid)
        dfi = filter(:jobid => (x)->x==jobid, df)
        sort!(dfi, :iteration)
        dfj = filter(:iteration => (x)->x>1, dfi)
        Δ = dfi.update_latency[1] + dfi.latency[1]
        Δ -= mean(dfj.update_latency) + mean(dfj.latency)
        df[df.jobid .== jobid, :time] .-= Δ
    end
    df
end

includet("analysis/convergence.jl")
includet("analysis/correlation.jl")
includet("analysis/latency.jl")
includet("analysis/simulation.jl")
includet("analysis/sweep.jl")
includet("analysis/netcoding.jl")
includet("analysis/newplots.jl")