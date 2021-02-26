using CodedComputing
using Random, MPI, HDF5, LinearAlgebra, SparseArrays
using Test

"""

Return an array composed of the PCA computed iterates.
"""
function load_pca_iterates(outputfile::AbstractString, outputdataset::AbstractString)
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test outputdataset in keys(fid)
        if "iterates" in keys(fid) && length(size(fid["iterates"])) == 3 && size(fid["iterates"], 3) > 0
            @test fid[outputdataset][:, :] ≈ fid["iterates"][:, :, end]
            return [fid["iterates"][:, :, i] for i in 1:size(fid["iterates"], 3)]
        else # return only the final iterate if intermediate iterates aren't stored
            return [fid[outpudataset][:, :]]
        end
    end
end

function test_pca_iterates(;X::AbstractMatrix, niterations::Integer, ncomponents::Integer, 
                            ev::Real, outputfile::AbstractString, outputdataset::AbstractString, atol=1e-2)
    dimension, nsamples = size(X)
    Vs = load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    @test length(Vs) == niterations
    @test all((V)->size(V)==(dimension, ncomponents), Vs)
    @test Vs[end]'*Vs[end] ≈ I
    @test isapprox(fs[end], ev, atol=atol)
    return Vs, fs
end

@testset "latency.jl" begin
    kernel = "../src/latency/kernel.jl"
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
    df = df_from_latency_file(outputfile)
    @test all(diff(df.timestamp) .>= timeout)
end

@testset "Linalg.jl" begin
    Random.seed!(123)
    n, m = 100, 10
    V = randn(n, m)
    orthogonal!(V)
    @test V'*V ≈ I
end

@testset "pca.jl" begin

    # setup
    Random.seed!(123)
    kernel = "../src/pca/pca.jl"
    nworkers = 2
    niterations = 200
    inputdataset = "X"
    outputdataset = "V"
    nsamples, dimension = 20, 10
    ncomponents = div(dimension, 2)

    # generate input dataset
    X = randn(dimension, nsamples)
    inputfile = tempname()
    h5open(inputfile, "w") do file
        file[inputdataset] = X
    end

    # correct solution (computed via LinearAlgebra.svd)
    V = pca(X, ncomponents)
    ev = explained_variance(X, V)

    ### exact
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --niterations $niterations 
        --ncomponents $ncomponents
        --saveiterates
        ```))
    Vs, _ = test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)

    ### providing an initial iterate
    iteratedataset = "V"
    niterations = 1
    h5open(inputfile, "r+") do fid
        fid[iteratedataset] = Vs[end]
    end
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --niterations $niterations
        --ncomponents $ncomponents
        --saveiterates
        --iteratedataset $iteratedataset
        ```))    
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)        
    
    ### exact with replication
    nreplicas = 2
    niterations = 200
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --niterations $niterations 
        --ncomponents $ncomponents
        --nreplicas $nreplicas
        --saveiterates
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)                

    ### replication + ignoring the slowest worker
    nworkers = 2
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $ncomponents
        --niterations $niterations 
        --nwait $(nworkers-1)
        --nreplicas $nreplicas        
        --saveiterates
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)

    ### 12 workers, a factor 3 replication
    nworkers = 12
    nreplicas = 3
    nwait = div(nworkers, nreplicas)
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --niterations $niterations 
        --ncomponents $ncomponents
        --nreplicas $nreplicas
        --nwait $nwait
        --saveiterates
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)                            

    ## using mini-batches
    nworkers = 2
    outputfile = tempname()
    pfraction = 0.9
    stepsize = pfraction
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $ncomponents
        --niterations $niterations 
        --nwait $(nworkers-1) 
        --pfraction $pfraction 
        --stepsize $stepsize
        --saveiterates
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset, atol=Inf)

    ### sub-partitioning and stochastic sub-gradients
    nsubpartitions = 2
    pfraction = 0.9
    stepsize = pfraction / nsubpartitions
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile
        --ncomponents $ncomponents
        --niterations $niterations 
        --nwait $(nworkers-1) 
        --pfraction $pfraction 
        --nsubpartitions $nsubpartitions 
        --stepsize $stepsize
        --saveiterates
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset, atol=Inf)
end

@time @testset "pca.jl (sparse matrices)" begin

    # setup
    Random.seed!(123)
    kernel = "../src/pca/pca.jl"
    nworkers = 2
    niterations = 200
    inputdataset = "X"
    outputdataset = "V"
    nsamples, dimension = 20, 10
    ncomponents = div(dimension, 2)
    p = 0.9 # matrix density    

    # generate input dataset
    X = sprand(dimension, nsamples, p)
    inputfile = tempname()
    h5writecsc(inputfile, inputdataset, X)

    # exact solution (computed via LinearAlgebra.svd)
    V = pca(Matrix(X), ncomponents)
    ev = explained_variance(X, V)

    # exact solution
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --niterations $niterations
        --ncomponents $ncomponents
        --saveiterates
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)        
end

@testset "pca.jl (variance reduced)" begin
    # with variance reduction, the algorithm will always converge to the optimum

    # setup
    Random.seed!(123)
    kernel = "../src/pca/pca.jl"
    nworkers = 2
    niterations = 100
    inputdataset = "X"
    outputdataset = "V"
    nsamples, dimension = 20, 10
    ncomponents = div(dimension, 2)

    # generate input dataset
    X = randn(dimension, nsamples)
    inputfile = tempname()
    h5open(inputfile, "w") do file
        file[inputdataset] = X
    end
    V = pca(X, ncomponents)    
    ev = explained_variance(X, V)    

    ### partitioning the dataset over the workers
    stepsize = 1
    outputfile = tempname()
    nsubpartitions = 1 
    mpiexec(cmd -> run(```$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $ncomponents
        --niterations $niterations 
        --stepsize $stepsize        
        --nsubpartitions $nsubpartitions
        --nwait $(nworkers-1)
        --variancereduced
        --saveiterates        
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)        

    ### with a factor 2 replication
    outputfile = tempname()
    nsubpartitions = 1
    nworkers = 4
    nreplicas = 2
    npartitions = div(nworkers, nreplicas)
    mpiexec(cmd -> run(```$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $ncomponents    
        --niterations $niterations 
        --stepsize $stepsize        
        --nsubpartitions $nsubpartitions
        --nwait $(npartitions-1)
        --variancereduced
        --saveiterates        
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)                
    
    ### same as the previous, but with kickstart enabled
    outputfile = tempname()    
    mpiexec(cmd -> run(```$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $ncomponents    
        --niterations $niterations 
        --stepsize $stepsize        
        --nsubpartitions $nsubpartitions
        --nwait $(npartitions-1)
        --variancereduced
        --kickstart
        --saveiterates        
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)                    
    
    ### with a factor 3 replication
    outputfile = tempname()
    nsubpartitions = 1
    nworkers = 3
    nreplicas = 3
    npartitions = div(nworkers, nreplicas)
    mpiexec(cmd -> run(```$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $ncomponents    
        --niterations $niterations 
        --stepsize $stepsize        
        --nsubpartitions $nsubpartitions
        --nwait $npartitions
        --variancereduced
        --saveiterates        
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)                            

    ### sub-partitioning the data stored at each worker    
    nworkers = 2
    niterations = 100
    nsubpartitions = 2
    stepsize = 1/2
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $ncomponents
        --niterations $niterations 
        --stepsize $stepsize        
        --nwait $(nworkers-1)        
        --nsubpartitions $nsubpartitions
        --variancereduced
        --saveiterates        
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)

    # with replication
    nworkers = 4
    nreplicas = 2
    npartitions = div(nworkers, nreplicas)
    niterations = 100
    stepsize = 1/2
    nsubpartitions = 2
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $ncomponents
        --niterations $niterations 
        --stepsize $stepsize        
        --nwait $(npartitions-1)
        --nsubpartitions $nsubpartitions
        --nreplicas $nreplicas
        --variancereduced
        --saveiterates        
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset)    
end

@testset "power.jl" begin
      
    # setup
    Random.seed!(123)
    kernel = "../src/pca/power.jl"
    nworkers = 2
    npartitions = nworkers
    niterations = 200
    inputdataset = "X"
    outputdataset = "V"
    n, m = 20, 10
    k = div(m, 2)

    # generate input dataset
    X = randn(n, m)
    inputfile = tempname()
    h5open(inputfile, "w") do file
        file[inputdataset] = X
    end
    V_exact = pca(X, k)
    ev_exact = explained_variance(X, V_exact)
    
    ### code weight 1 (i.e., coding is equivalent to multiplying by a scalar)
    ### the computed results can be recovered exactly
    codeweight = 1
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile
        --npartitions $npartitions 
        --codeweight $codeweight 
        --niterations $niterations 
        --ncomponents $ncomponents
        --saveiterates
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    # println("SGD convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)    
    @test Vs[end]'*Vs[end] ≈ I
    @test isapprox(fs[end], ev_exact, atol=1e-2)    

    ### providing an initial iterate
    iteratedataset = "V"
    niterations = 1
    h5open(inputfile, "r+") do fid
        fid[iteratedataset] = Vs[end]
    end
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile
        --npartitions $npartitions 
        --codeweight $codeweight 
        --niterations $niterations 
        --ncomponents $ncomponents
        --saveiterates
        --iteratedataset $iteratedataset
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]    
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)
    @test Vs[end]'*Vs[end] ≈ I
    @test isapprox(fs[end], ev_exact, atol=1e-2)      
    
    ### code weight 2
    ### the computed results can be recovered exactly    
    codeweight = 2
    niterations = 200
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile
        --npartitions $npartitions
        --codeweight $codeweight
        --niterations $niterations
        --ncomponents $ncomponents
        --saveiterates
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    # println("SGD convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)    
    @test Vs[end]'*Vs[end] ≈ I
    @test isapprox(fs[end], ev_exact, atol=1e-2)

    ### code weight 1, nwait 1
    ### the computed results can be recovered exactly        
    codeweight = 1
    nwait = nworkers - 1
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile
        --nwait $nwait
        --npartitions $npartitions
        --codeweight $codeweight
        --niterations $niterations
        --ncomponents $ncomponents
        --saveiterates
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    # println("SGD convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)    
    @test Vs[end]'*Vs[end] ≈ I
    @test isapprox(fs[end], ev_exact, atol=1e-2)    

    ### code weight 2, nwait 1
    ### the computed results can't be recovered exactly            
    codeweight = 2
    nwait = nworkers - 1
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile
        --nwait $nwait
        --npartitions $npartitions
        --codeweight $codeweight
        --niterations $niterations
        --ncomponents $ncomponents
        --saveiterates
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    # println("SGD convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)    
    @test Vs[end]'*Vs[end] ≈ I
    @test isapprox(fs[end], ev_exact, atol=1e-2)         
end

@testset "HDF5Sparse.jl" begin
    Random.seed!(123)
    m, n, p = 10, 5, 0.1
    M = sprand(Float32, m, n, p)
    filename = tempname()
    name = "M"
    h5writecsc(filename, name, M)
    M_hat = h5readcsc(filename, name)
    @test typeof(M_hat) == typeof(M)
    @test M_hat ≈ M
end