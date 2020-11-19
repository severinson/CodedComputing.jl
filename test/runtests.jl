using CodedComputing
using MPI, HDF5, LinearAlgebra
using Test

@testset "CodedComputing.jl" begin
    # Write your tests here.
end

@testset "Principal component analysis" begin

    # setup
    kernel = "../src/pca/pca.jl"
    nworkers = 3
    niterations = 200
    inputdataset = "X"
    outputdataset = "V"    
    n, m = 20, 10
    k = m    

    # generate input dataset
    X = randn(n, m)
    inputfile = tempname()
    outputfile = tempname()
    h5open(inputfile, "w") do file
        file[inputdataset] = X
    end

    # run the kernel
    mpiexec(cmd -> run(`$cmd -n $nworkers julia --project $kernel $inputfile $outputfile --niterations $niterations`))
    V_correct = pca(X, k)
    V = similar(V_correct)

    # test that the output was generated correctly
    @test isfile(outputfile)
    h5open(outputfile, "r") do file
        @test outputdataset in names(file)
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

    # print benchmark data (if available)
    h5open(outputfile, "r") do file
        for name in ["ts_compute", "ts_update"]
            if name in names(file)
                println(name*":")
                println(file[name][:])
                println()
            end
        end
    end
end