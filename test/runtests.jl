using CodedComputing
using MPI, JLD, LinearAlgebra
using Test

@testset "CodedComputing.jl" begin
    # Write your tests here.
end

# mpitestexec(n::Integer, file::String) = mpiexec(cmd -> run(`$cmd -n $n --mca btl ^openib julia --project $file`))

@testset "Principal component analysis" begin

    # setup
    kernel = "../src/pca/pca.jl"
    nworkers = 3
    niterations = 100
    inputdataset = "X"
    outputdataset = "V"    
    n, m = 20, 10
    k = m    

    # generate input dataset
    X = randn(n, m)
    path, _ = mktemp()
    path = tempname()
    jldopen(path, "w", compress=true) do file
        file[inputdataset] = X
    end

    # run the kernel
    # mpiexec(cmd -> run(`$cmd -n $nworkers --mca btl ^openib julia --project $kernel $path $inputdataset $path $outputdataset $k $niterations`))
    mpiexec(cmd -> run(`$cmd -n $nworkers --mca btl ^openib julia --project $kernel $path --niterations $niterations --benchmarkfile $path`))
    V_correct = pca(X, k)
    V = similar(V_correct)

    # test that the output was generated correctly
    jldopen(path, "r") do file
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
    jldopen(path, "r") do file
        for name in ["ts_compute", "ts_update"]
            if name in names(file)
                println(name*":")
                println(file[name][:])
                println()
            end
        end
    end
end