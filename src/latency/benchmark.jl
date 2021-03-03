using CSV, DataFrames, SparseArrays

export latency_benchmark

function latency_benchmark()
    kernel = "src/latency/kernel.jl"
    nwait = 1
    nworkers = 3
    niterations = 10
    timeout = 0.1
    nbytes = 100
    nrows = 100
    ncols = 100
    ncomponents = 3
    density = 0.1
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $outputfile
            --niterations $niterations
            --nbytes $nbytes
            --nrows $nrows
            --ncols $ncols
            --ncomponents $ncomponents
            --density $density
            --nwait $nwait
            --timeout $timeout            
    ```))
    return df_from_latency_file(outputfile)
end

function sample!(V, W, X)
    mul!(W, X, V)
    mul!(V, X', W)
end

function samples(X, V; nsamples::Integer)
    W = similar(V, size(X, 1), size(V, 2))
    timestamps = zeros(nsamples)
    latencies = zeros(nsamples)

    # force JIT    
    t0 = time_ns()
    timestamps[1] = (time_ns() - t0) / 1e9
    latencies[1] = @elapsed sample!(V, W, X)

    # take samples
    t0 = time_ns()    
    for i in 1:nsamples
        timestamps[i] = (time_ns() - t0) / 1e9
        latencies[i] = @elapsed sample!(V, W, X)
    end
    timestamps, latencies
end

function latency_sweep()

    ncols = 1000
    ncomponents = 3
    density = 0.05
    nsamples = 100

    df = DataFrame()
    V = randn(ncols, ncomponents)

    # dense matrices    
    nrows_all = [1000, 2000, 4000, 8000, 16000, 32000]    
    for nrows in nrows_all    
        dfi = DataFrame()
        Xd = randn(nrows, ncols)
        timestamps, latencies = samples(Xd, V; nsamples)
        dfi.timestamp = timestamps
        dfi.latency = latencies
        dfi.nrows = nrows
        dfi.ncols = ncols
        dfi.ncomponents = ncomponents
        dfi.density = 1.0        
        df = vcat(df, dfi, cols=:union)
    end

    # sparse matrices    
    nrows_all = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    for nrows in nrows_all            
        dfi = DataFrame()        
        Xs = sprand(nrows, ncols, density)
        timestamps, latencies = samples(Xs, V; nsamples)        
        dfi.timestamp = timestamps
        dfi.latency = latencies
        dfi.nrows = nrows
        dfi.ncols = ncols
        dfi.ncomponents = ncomponents
        dfi.density = density
        df = vcat(df, dfi, cols=:union)        
    end

    df.nflops = 2 .* df.nrows .* df.ncols .* df.ncomponents .* df.density
    df
end