using CodedComputing
using Random, MPI, HDF5, LinearAlgebra, SparseArrays
using Test

@testset "Linalg.jl" begin
    Random.seed!(123)
    n, m = 100, 10
    V = randn(n, m)
    orthogonal!(V)
    @test V'*V ≈ I
end

function test_load_pca_iterates(outputfile::AbstractString, name::AbstractString)
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test name in keys(fid)
        if "iterates" in keys(fid) && length(size(fid["iterates"])) == 3 && size(fid["iterates"], 3) > 0
            @test fid[name][:, :] ≈ fid["iterates"][:, :, end]
            return [fid["iterates"][:, :, i] for i in 1:size(fid["iterates"], 3)]
        else
            return [fid[name][:, :]]
        end
    end
end

@testset "pca.jl" begin

    # setup
    Random.seed!(123)
    kernel = "../src/pca/pca.jl"
    nworkers = 2
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

    # correct solution (computed via LinearAlgebra.svd)
    V_exact = pca(X, k)
    ev_exact = explained_variance(X, V_exact)
    println("Exact explained variance: $ev_exact")

    ### exact
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --niterations $niterations 
        --ncomponents $k 
        --saveiterates
        ```))    
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]    
    # println("SGD (exact) convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)
    @test Vs[end]'*Vs[end] ≈ I
    @test isapprox(fs[end], ev_exact, atol=1e-2)
    
    ### exact with replication
    nreplicas = 2
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --niterations $niterations 
        --ncomponents $k
        --nreplicas $nreplicas
        --saveiterates
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    # println("SGD (exact) convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)    
    @test Vs[end]'*Vs[end] ≈ I
    @test isapprox(fs[end], ev_exact, atol=1e-2)

    ### ignoring the slowest worker
    nworkers = 2
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $k 
        --niterations $niterations 
        --nwait $(nworkers-1)
        --saveiterates
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    # println("SGD convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)    
    @test Vs[end]'*Vs[end] ≈ I

    ## using mini-batches
    outputfile = tempname()
    pfraction = 0.9
    stepsize = pfraction
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $k 
        --niterations $niterations 
        --nwait $(nworkers-1) 
        --pfraction $pfraction 
        --stepsize $stepsize
        --saveiterates
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    # println("SGD convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)    
    @test Vs[end]'*Vs[end] ≈ I

    ### sub-partitioning and stochastic sub-gradients
    nsubpartitions = 2
    pfraction = 0.9
    stepsize = pfraction / nsubpartitions
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile
        --ncomponents $k 
        --niterations $niterations 
        --nwait $(nworkers-1) 
        --pfraction $pfraction 
        --nsubpartitions $nsubpartitions 
        --stepsize $stepsize
        --saveiterates
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    # println("SGD convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)    
    @test Vs[end]'*Vs[end] ≈ I
end

@time @testset "pca.jl (sparse matrices)" begin

    # setup
    Random.seed!(123)
    kernel = "../src/pca/pca.jl"
    nworkers = 2
    niterations = 200
    inputdataset = "X"
    outputdataset = "V"
    n, m = 20, 10
    k = div(m, 2)
    p = 0.9 # matrix density    

    # generate input dataset
    X = sprand(n, m, p)
    inputfile = tempname()
    h5writecsc(inputfile, inputdataset, X)

    # exact solution (computed via LinearAlgebra.svd)
    V_exact = pca(Matrix(X), k)
    ev_exact = explained_variance(X, V_exact)

    # exact solution
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $k 
        --niterations $niterations 
        --ncomponents $k
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

@testset "pca.jl (variance reduced)" begin
    # with variance reduction, the algorithm should always converge eventually

    # setup
    Random.seed!(123)
    kernel = "../src/pca/pca.jl"
    nworkers = 2
    niterations = 100
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

    ### partitioning the dataset over the workers
    stepsize = 1
    outputfile = tempname()
    nsubpartitions = 1 
    mpiexec(cmd -> run(```$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $k    
        --niterations $niterations 
        --stepsize $stepsize        
        --nsubpartitions $nsubpartitions
        --nwait $(nworkers-1)
        --variancereduced
        --saveiterates        
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    # println("SGD convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)    
    @test Vs[end]'*Vs[end] ≈ I
    @test isapprox(fs[end], ev_exact, atol=1e-2)                

    ### sub-partitioning the data stored at each worker
    niterations = 100
    stepsize = 1/2
    nsubpartitions = 2
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $k
        --niterations $niterations 
        --stepsize $stepsize        
        --nwait $(nworkers-1)        
        --nsubpartitions $nsubpartitions
        --variancereduced
        --saveiterates        
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    # println("SGD convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)    
    @test Vs[end]'*Vs[end] ≈ I
    @test isapprox(fs[end], ev_exact, atol=1e-2)                        

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
        --ncomponents $k
        --niterations $niterations 
        --stepsize $stepsize        
        --nwait $(npartitions-1)
        --nsubpartitions $nsubpartitions
        --nreplicas $nreplicas
        --variancereduced
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
        --ncomponents $k
        --saveiterates
        ```))
    Vs = test_load_pca_iterates(outputfile, outputdataset)
    fs = [explained_variance(X, V) for V in Vs]
    # println("SGD convergence: $fs")
    @test length(Vs) == niterations
    @test all((V)->size(V)==(m,k), Vs)    
    @test Vs[end]'*Vs[end] ≈ I
    @test isapprox(fs[end], ev_exact, atol=1e-2)    
    
    ### code weight 2
    ### the computed results can be recovered exactly    
    codeweight = 2
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile
        --npartitions $npartitions
        --codeweight $codeweight
        --niterations $niterations
        --ncomponents $k
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
        --ncomponents $k
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
        --ncomponents $k
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