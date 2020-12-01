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
    for i in 1:k
        @test isapprox(
            CodedComputing.minangle(view(V, :, i), view(V_correct, :, i)),
            0, atol=1e-2
        )
    end

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
    nminibatches = 2
    stepsize = 1/nminibatches
    mpiexec(cmd -> run(`$cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile --niterations $niterations --nwait $(nworkers-1) --nminibatches $nminibatches --stepsize $stepsize`))

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

end

@testset "pcacsc.jl" begin

    # setup
    Random.seed!(123)
    kernel = "../src/pca/pcacsc.jl"
    nworkers = 2
    niterations = 200
    inputdataset = "X"
    outputdataset = "V"
    n, m = 20, 10
    k = m
    p = 0.9 # matrix density

    # generate input dataset
    X = sprand(n, m, p)
    inputfile = tempname()
    h5writecsc(inputfile, inputdataset, X)

    # correct solution (computed via LinearAlgebra.svd)
    V_correct = pca(Matrix(X), k)
    V = similar(V_correct)

    ### exact
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
    for i in 1:k
        @test isapprox(
            CodedComputing.minangle(view(V, :, i), view(V_correct, :, i)),
            0, atol=1e-2
        )
    end
end

@testset "pcasega.jl" begin

    # setup
    Random.seed!(123)
    kernel = "../src/pca/pcasega.jl"
    nworkers = 2
    niterations = 200
    inputdataset = "X"
    outputdataset = "V"
    n, m = 20, 10
    k = m
    p = 0.9 # matrix density

    # generate input dataset
    X = sprand(n, m, p)
    inputfile = tempname()
    h5writecsc(inputfile, inputdataset, X)

    # correct solution (computed via LinearAlgebra.svd)
    V_correct = pca(Matrix(X), k)
    V = similar(V_correct)

    ### exact
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
    for i in 1:k
        @test isapprox(
            CodedComputing.minangle(view(V, :, i), view(V_correct, :, i)),
            0, atol=1e-2
        )
    end
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