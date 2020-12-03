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

@testset "pca.jl" begin

    # setup
    Random.seed!(123)
    kernel = "../src/pca/pca.jl"
    nworkers = 2
    niterations = 200
    inputdataset = "X"
    outputdataset = "V"
    n, m = 20, 10
    k = m    

    # generate input dataset
    X = randn(n, m)
    inputfile = tempname()
    h5open(inputfile, "w") do file
        file[inputdataset] = X
    end

    # correct solution (computed via LinearAlgebra.svd)
    V_correct = pca(X, k)
    V = similar(V_correct)
    ev_correct = explained_variance(X, V_correct)

    ### exact
    outputfile = tempname()
    mpiexec(cmd -> run(`$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile --niterations $niterations --ncomponents $k`))

    # test that the output was generated correctly
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do file
        @test outputdataset in keys(file)
        @test size(file[outputdataset]) == (m, k)
        V .= file[outputdataset][:, :]
    end

    # test that the columns are orthogonal
    @test V'*V ≈ I

    # compare the computed principal components with those obtained from the built-in svd
    @test explained_variance(X, V) ≈ ev_correct

    ### ignoring the slowest worker
    outputfile = tempname()
    mpiexec(cmd -> run(`$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile --niterations $niterations --nwait $(nworkers-1)`))

    # test that the output was generated correctly
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test outputdataset in keys(fid)
        @test size(fid[outputdataset]) == (m, k)
        V .= fid[outputdataset][:, :]
        @test length(fid["benchmark/t_compute"]) == niterations
    end

    # test that the columns are orthogonal
    @test V'*V ≈ I

    ## using mini batches
    outputfile = tempname()
    pfraction = 0.9
    stepsize = pfraction
    mpiexec(cmd -> run(`$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile --niterations $niterations --nwait $(nworkers-1) --pfraction $pfraction --stepsize $stepsize`))

    # test that the output was generated correctly
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test outputdataset in keys(fid)
        @test size(fid[outputdataset]) == (m, k)
        V .= fid[outputdataset][:, :]
        @test length(fid["benchmark/t_compute"]) == niterations
    end

    # test that the columns are orthogonal
    @test V'*V ≈ I

    ### using mini batches
    outputfile = tempname()
    pfraction = 0.9
    stepsize = pfraction
    mpiexec(cmd -> run(`$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile --niterations $niterations --nwait $(nworkers-1) --pfraction $pfraction --stepsize $stepsize`))

    # test that the output was generated correctly
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test outputdataset in keys(fid)
        @test size(fid[outputdataset]) == (m, k)
        V .= fid[outputdataset][:, :]
        @test length(fid["benchmark/t_compute"]) == niterations
    end

    # test that the columns are orthogonal
    @test V'*V ≈ I    

    ### sub-partitioning and stochastic sub-gradients
    nsubpartitions = 2
    pfraction = 0.9
    stepsize = pfraction / nsubpartitions
    outputfile = tempname()
    mpiexec(cmd -> run(`$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile --niterations $niterations --nwait $(nworkers-1) --pfraction $pfraction --nsubpartitions $nsubpartitions --stepsize $stepsize`))

    # test that the output was generated correctly
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test outputdataset in keys(fid)
        @test size(fid[outputdataset]) == (m, k)
        V .= fid[outputdataset][:, :]
        @test length(fid["benchmark/t_compute"]) == niterations
    end

    # test that the columns are orthogonal
    @test V'*V ≈ I
    
    ### variance reduction
    niterations = 100
    stepsize = 1
    outputfile = tempname()    
    mpiexec(cmd -> run(```$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --niterations $niterations 
        --stepsize $stepsize        
        --nwait $(nworkers-1)        
        --variancereduced
        ```))

    # test that the output was generated correctly
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test outputdataset in keys(fid)
        @test size(fid[outputdataset]) == (m, k)
        V .= fid[outputdataset][:, :]
        @test length(fid["benchmark/t_compute"]) == niterations
    end

    # test that the columns are orthogonal
    @test V'*V ≈ I

    # compare the computed principal components with those obtained from the built-in svd
    # with variance reduction, the algorithm should always converge eventually
    @test explained_variance(X, V) ≈ ev_correct

    ### sub-partitioning + variance reduction
    niterations = 100
    stepsize = 1
    nsubpartitions = 2
    outputfile = tempname()
    mpiexec(cmd -> run(```$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile 
        --niterations $niterations 
        --stepsize $stepsize        
        --nwait $(nworkers-1)        
        --nsubpartitions $nsubpartitions
        --variancereduced
        ```))

    # test that the output was generated correctly
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test outputdataset in keys(fid)
        @test size(fid[outputdataset]) == (m, k)
        V .= fid[outputdataset][:, :]
        @test length(fid["benchmark/t_compute"]) == niterations
    end

    # test that the columns are orthogonal
    @test V'*V ≈ I

    # compare the computed principal components with those obtained from the built-in svd
    # with variance reduction, the algorithm should always converge eventually
    @test explained_variance(X, V) ≈ ev_correct

    ### sparse (CSC) matrices
    p = 0.9 # matrix density    
    X = sprand(n, m, p)
    inputfile = tempname()
    h5writecsc(inputfile, inputdataset, X)

    # correct solution (computed via LinearAlgebra.svd)
    V_correct = pca(Matrix(X), k)
    V = similar(V_correct)

    # exact solution
    outputfile = tempname()
    mpiexec(cmd -> run(`$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile --niterations $niterations --ncomponents $k`))

    # test that the output was generated correctly
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test outputdataset in keys(fid)
        @test size(fid[outputdataset]) == (m, k)
        V .= fid[outputdataset][:, :]
        @test length(fid["benchmark/t_compute"]) == niterations
    end

    # test that the columns are orthogonal
    @test V'*V ≈ I
    
    # compare the computed principal components with those obtained from the built-in svd
    @test explained_variance(X, V) ≈ ev_correct    

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