export latency_benchmark

function latency_benchmark()
    kernel = "src/latency/latency.jl"
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