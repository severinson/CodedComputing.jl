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
            return [fid[outputdataset][:, :]]
        end
    end
end

function load_logreg_iterates(outputfile::AbstractString, outputdataset::AbstractString)
    @test HDF5.ishdf5(outputfile)
    h5open(outputfile, "r") do fid
        @test outputdataset in keys(fid)
        if "iterates" in keys(fid) && length(size(fid["iterates"])) == 2 && size(fid["iterates"], 2) > 0
            @test fid[outputdataset][:] ≈ fid["iterates"][:, end]
            return [fid["iterates"][:, i] for i in 1:size(fid["iterates"], 2)]
        else # return only the final iterate if intermediate iterates aren't stored
            return [fid[outputdataset][:]]
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
    @test fs[end] < ev || isapprox(fs[end], ev, atol=atol)
    return Vs, fs
end

@testset "Linalg.jl" begin
    Random.seed!(123)
    n, m = 100, 10
    V = randn(n, m)
    orthogonal!(V)
    @test V'*V ≈ I
end

# @testset "HDF5Sparse.jl" begin
#     Random.seed!(123)

#     # test writing and reading a sparse matrix
#     m, n, p = 10, 6, 0.5
#     M = sprand(Float32, m, n, p)
#     filename = tempname()
#     name = "M"
#     h5writecsc(filename, name, M, batchsize=5)
#     M_hat = h5readcsc(filename, name)
#     @test typeof(M_hat) == typeof(M)
#     @test M_hat ≈ M

#     # test reading each column separately
#     for i in 1:n
#         v = h5readcsc(filename, name, i)
#         @test v ≈ M[:, i]
#     end

#     # test reading blocks of columns
#     for i in 1:n
#         for j in i:n
#             correct = M[:, i:j]
#             V = h5readcsc(filename, name, i, j)
#             @test V ≈ M[:, i:j]
#         end
#     end

#     # test appending more columns to a matrix
#     M2 = sprand(Float32, m, n, p)
#     h5appendcsc(filename, name, M2)
#     M_hat = h5readcsc(filename, name)
#     @test typeof(M_hat) == typeof(M)
#     @test M_hat ≈ hcat(M, M2)

#     # test out-of-core column-wise block permutation
#     M = sprand(10, 20, 0.5)
#     h5writecsc(filename, "M", M, overwrite=true)
#     p = [1, 2, 3]
#     h5permutecsc(filename, "M", filename, "Mp", p, overwrite=true)
#     Mp = h5readcsc(filename, "Mp")
#     @test Mp ≈ M

#     p = [2, 1]
#     h5permutecsc(filename, "M", filename, "Mp", p, overwrite=true)
#     Mp = h5readcsc(filename, "Mp")
#     @test Mp[:, 1:10] ≈ M[:, 11:20]
#     @test Mp[:, 11:20] ≈ M[:, 1:10]

#     p = randperm(size(M, 2))
#     h5permutecsc(filename, "M", filename, "Mp", p, overwrite=true)
#     Mp = h5readcsc(filename, "Mp")
#     @test Mp ≈ M[:, p]

#     for n in [20, 21, 23, 23]
#         M = sprand(10, n, 0.5)
#         h5writecsc(filename, "M", M, overwrite=true)
#         p = randperm(4)
#         h5permutecsc(filename, "M", filename, "Mp", p, overwrite=true)
#         Mp = h5readcsc(filename, "Mp")
#         @test size(Mp) == size(M)
#     end

#     # test partially out-of-core column-wise permutation
#     for n in [20, 21, 23, 23]
#         M = sprand(10, n, 0.5)
#         p = randperm(n)
#         h5permutecsc(M, filename, "Mp", p, nblocks=4, overwrite=true)
#         Mp = h5readcsc(filename, "Mp")
#         @test Mp ≈ M[:, p]
#     end

#     # test matrix-matrix multiplication
#     for n in [20, 21, 23, 23, 24]
#         M = sprand(10, n, 0.5)
#         h5writecsc(filename, "M", M, overwrite=true)        
#         A = randn(20, 10)
#         C = h5mulcsc(A, filename, "M", nblocks=4)
#         @test C ≈ A*M
#     end
# end

# @testset "latency.jl" begin
#     kernel = "../src/latency/kernel.jl"
#     nwait = 1
#     nworkers = 3
#     niterations = 10
#     timeout = 0.1
#     nbytes = 100
#     nrows = 100
#     ncols = 100
#     ncomponents = 3
#     density = 0.1
#     outputfile = tempname()
#     mpiexec(cmd -> run(```
#         $cmd -n $(nworkers+1) julia --project $kernel $outputfile
#             --niterations $niterations
#             --nbytes $nbytes
#             --nrows $nrows
#             --ncols $ncols
#             --ncomponents $ncomponents
#             --density $density
#             --nwait $nwait
#             --timeout $timeout            
#     ```))
#     df = df_from_latency_file(outputfile)
#     @test all(diff(df.timestamp) .>= timeout)
# end

function logreg_loss(v, X, b, λ)
    rv = 0.0
    for i in 1:length(b)
        rv += log(1 + exp(-b[i]*(v[1]+dot(X[:, i], view(v, 2:length(v))))))
    end
    rv / length(b) + λ/2 * norm(v)^2
end

@testset "logreg.jl" begin
    kernel = "../src/pca/logreg.jl"    

    # input data set with known optimal solution
    X = [1.444786643000158 0.49236792885913283 -0.53258473265429 0.05476455630673194 -1.3473893605265843; 0.48932299731783646 2.0708445447107926 1.2414596020757043 0.9131934117095984 -0.15692043560721075; 0.7774625331093794 0.7234405608945721 -0.037446104354257874 -1.1104987697394342 1.354975413199728]
    b = [-1, 1, -1, -1, 1]
    v_opt = [-0.3423591553493419, -0.41317049965033387, 0.007294166575956451, 0.6763846515861628]
    m, n = size(X)
    λ = 1 / n
    opt = logreg_loss(v_opt, X, b, λ)

    # write test problem to disk
    inputfile = tempname()    
    inputdataset = "X"
    outputdataset = "V"
    labeldataset = "b"
    h5open(inputfile, "w") do file
        file[inputdataset] = X
        file[labeldataset] = b
    end

    # GD
    nworkers = 2
    niterations = 100
    stepsize = 0.1
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile        
        --inputdataset $inputdataset
        --outputdataset $outputdataset
        --niterations $niterations
        --saveiterates
        --lambda $λ
        ```))
    vs = load_logreg_iterates(outputfile, outputdataset)
    v = vs[end]
    f = logreg_loss(v, X, b, λ)
    @test f < opt || isapprox(f, opt, rtol=1e-2)

    # DSAG
    nworkers = 2
    nwait = 1
    niterations = 100
    stepsize = 0.1
    nsubpartitions = 2
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile        
        --inputdataset $inputdataset
        --nwait $nwait
        --variancereduced
        --nsubpartitions $nsubpartitions
        --outputdataset $outputdataset
        --niterations $niterations
        --saveiterates
        --lambda $λ
        ```))
    vs = load_logreg_iterates(outputfile, outputdataset)
    v = vs[end]
    f = logreg_loss(v, X, b, λ)    
    @test f < opt || isapprox(f, opt, rtol=1e-2)

    # DSAG w. nwaitschedule < 1.0
    nworkers = 2
    nwait = 2
    niterations = 100
    stepsize = 0.1
    nsubpartitions = 2
    nwaitschedule = 0.9
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile        
        --inputdataset $inputdataset
        --nwait $nwait
        --variancereduced
        --nsubpartitions $nsubpartitions
        --outputdataset $outputdataset
        --niterations $niterations
        --saveiterates
        --lambda $λ
        --nwaitschedule $nwaitschedule
        ```))
    vs = load_logreg_iterates(outputfile, outputdataset)
    v = vs[end]
    f = logreg_loss(v, X, b, λ)    
    @test f < opt || isapprox(f, opt, rtol=1e-2)    

    # DSAG w. sparse input data
    X = sparse(X)
    inputfile = tempname()
    h5writecsc(inputfile, inputdataset, X)
    h5open(inputfile, "cw") do file
        file[labeldataset] = b
    end

    nworkers = 2
    nwait = 1
    niterations = 100
    stepsize = 0.1
    nsubpartitions = 2
    outputfile = tempname()
    mpiexec(cmd -> run(```
        $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile        
        --inputdataset $inputdataset
        --nwait $nwait
        --variancereduced
        --nsubpartitions $nsubpartitions
        --outputdataset $outputdataset
        --niterations $niterations
        --saveiterates
        --lambda $λ
        ```))
    vs = load_logreg_iterates(outputfile, outputdataset)
    v = vs[end]
    f = logreg_loss(v, X, b, λ)        
    @test f < opt || isapprox(f, opt, rtol=1e-2)    
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