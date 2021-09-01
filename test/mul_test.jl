using Random, SparseArrays

@testset "SparseMatrixCSC with matrices" begin
    rng = MersenneTwister(123)
    A = sprand(rng, 10, 5, 0.2)
    C = zeros(10, 2)
    B = randn(5, 2)

    cols = 1:5
    colsmul!(C, A, B, cols)
    @test C ≈ A[:, cols]*B[cols, :]

    cols = 2:4
    colsmul!(C, A, B, cols)
    @test C ≈ A[:, cols]*B[cols, :]

    cols = [1, 5]
    colsmul!(C, A, B, cols)
    @test C ≈ A[:, cols]*B[cols, :]

    α, β = rand(), rand()
    Cc = copy(C)
    colsmul!(C, A, B, cols, α, β)
    @test C ≈ α.*A[:, cols]*B[cols, :] .+ β.*Cc
end

@testset "Transposed SparseMatrixCSC with matrices" begin
    rng = MersenneTwister(123)
    A = sprand(rng, 10, 5, 0.2)
    # At = Transpose(A)
    C = zeros(5, 2)
    B = randn(10, 2)

    cols = 1:5
    tcolsmul!(C, A, B, cols)
    @test C[cols, :] ≈ A[:, cols]'*B

    cols = 2:4
    tcolsmul!(C, A, B, cols)
    @test C[cols, :] ≈ A[:, cols]'*B

    cols = [1, 5]
    tcolsmul!(C, A, B, cols)
    @test C[cols, :] ≈ A[:, cols]'*B

    α, β = rand(), rand()
    Cc = copy(C)
    tcolsmul!(C, A, B, cols, α, β)
    @test C[cols, :] ≈ α.*A[:, cols]'*B .+ β.*Cc[cols, :]
end

@testset "SparseMatrixCSC with vectors" begin
    rng = MersenneTwister(123)
    A = sprand(rng, 10, 5, 0.2)
    c = zeros(10)
    b = randn(5)

    cols = 1:5
    colsmul!(c, A, b, cols)
    @test c ≈ A[:, cols]*b[cols]

    cols = 2:4
    colsmul!(c, A, b, cols)
    @test c ≈ A[:, cols]*b[cols]

    cols = [1, 5]
    colsmul!(c, A, b, cols)
    @test c ≈ A[:, cols]*b[cols]

    α, β = rand(), rand()
    cc = copy(c)
    colsmul!(c, A, b, cols, α, β)
    @test c ≈ α.*A[:, cols]*b[cols] .+ β.*cc
end

@testset "Matrix with matrices" begin
    A = randn(10, 5)
    C = zeros(10, 2)
    B = randn(5, 2)

    cols = 1:5
    colsmul!(C, A, B, cols)
    @test C ≈ A[:, cols]*B[cols, :]

    cols = 2:4
    colsmul!(C, A, B, cols)
    @test C ≈ A[:, cols]*B[cols, :]

    cols = [1, 5]
    colsmul!(C, A, B, cols)
    @test C ≈ A[:, cols]*B[cols, :]

    α, β = rand(), rand()
    Cc = copy(C)
    colsmul!(C, A, B, cols, α, β)
    @test C ≈ α.*A[:, cols]*B[cols, :] .+ β.*Cc
end
