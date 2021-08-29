@testset "gd/sgd" begin

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
    stepsize = 1.0
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --ncomponents $ncomponents
        --niterations $niterations 
        --nwait $(nworkers-1) 
        --stepsize $stepsize
        --saveiterates
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset, atol=Inf)

    ### sub-partitioning and stochastic sub-gradients
    nsubpartitions = 2
    stepsize = 1.0 / nsubpartitions
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile
        --ncomponents $ncomponents
        --niterations $niterations 
        --nwait $(nworkers-1) 
        --nsubpartitions $nsubpartitions 
        --stepsize $stepsize
        --saveiterates
        ```))
    test_pca_iterates(;X, niterations, ncomponents, ev, outputfile, outputdataset, atol=Inf)
end

@time @testset "sparse matrices" begin

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
    !isfile(inputfile) || rm(inputfile)
    # fid = h5open(inputfile, "cw")
    
    # # h5writecsc(inputfile, inputdataset, X)
    A = H5SparseMatrixCSC(inputfile, inputdataset, X)
    flush(A.fid)

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

@time @testset "variance-reduced" begin
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

    ### same as the previous, but with nwaitschedule < 1
    nwaitschedule = 0.9
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
        --nwaitschedule $nwaitschedule
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