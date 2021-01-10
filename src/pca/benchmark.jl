using HDF5, MPI, Dates
using CodedComputing

include("../encoding/diggavi.jl")
include("../analysis/pca.jl")

function test_matrix(n, m, k)
    G = randn(n, m)
    P = orthogonal!(randn(m, k))
    X = (G*P)*P'
    X .+= randn(size(G)...)./100
    X
end

function write_input(n, m, k; inputfile::AbstractString, name="X")

    # don't do anything if the input file already exists
    if isfile(inputfile)
        @assert HDF5.ishdf5(inputfile)
        return h5read(inputfile, name)[:, :]
    end

    # generate a random input data matrix if it doesn't already exist
    if k == m
        X = randn(n, m)
    else
        X = test_matrix(n, m, k)
    end
    h5write(inputfile, name, X)
    return X
end

function write_coded_input(X; inputfile::AbstractString, name="C", coderate=1)
    nc = ceil(Int, size(X, 1) / coderate) # number of coded rows
    C = Matrix(encode_hadamard(X, nc))
    h5write(inputfile, name, C)
    return C
end

function benchmark_main(n=5000, m=5000, k=5000, directory=joinpath("./simulations", "$(now())"), coderate=1)

    # (1000, 500, 100)
    # directory = "/home/albin/.julia/dev/CodedComputing.jl/simulations/2020-12-17T14:19:57.949"
    
    # (5000, 5000, 200)
    # directory = "/home/albin/.julia/dev/CodedComputing.jl/simulations/2020-12-17T14:20:23.572"

    # prepare a directory for storing everything
    inputfile = joinpath(directory, "inputfile.h5")
    mkpath(directory)

    # simulation parameters
    kernel = "./src/pca/pca.jl"
    inputdataset = "X"
    nworkers = 47
    ncomponents = k
    niterations = 10

    X = write_input(n, m, k; inputfile, name=inputdataset)
    if coderate < 1
        write_coded_input(X; inputfile, name="C")
        inputdataset = "C"
    end

    # Measure iteration time
    stepsize = 0.9
    for nwait in round.(Int, range(1, nworkers, length=5))
        for nsubpartitions in [1, 3, 5]
            for ncomponents in [1, 100, 200]
                println((nwait, nsubpartitions, ncomponents))

                # with variance reduction
                outputfile = joinpath(directory, "output_$(now()).h5")
                mpiexec(cmd -> run(```
                    $cmd -n $(nworkers+1) julia --project $kernel 
                    $inputfile $outputfile 
                    --niterations $niterations
                    --nwait $nwait
                    --ncomponents $ncomponents
                    --stepsize $stepsize
                    --nsubpartitions $nsubpartitions
                    --variancereduced
                    --saveiterates
                    ```))

                # without variance reduction
                outputfile = joinpath(directory, "output_$(now()).h5")
                mpiexec(cmd -> run(```
                    $cmd -n $(nworkers+1) julia --project $kernel 
                    $inputfile $outputfile 
                    --niterations $niterations 
                    --nwait $nwait
                    --ncomponents $ncomponents
                    --stepsize $stepsize
                    --nsubpartitions $nsubpartitions
                    --saveiterates
                    ```))  
            end
        end
    end

    # # substitute decoding power method
    # kernel = "./src/pca/power.jl"
    # for nwait in [1, 10, 11, 12]
    #     for npartitions in [4, 6]
    #         for ncomponents in [1, 100]
    #             outputfile = joinpath(directory, "output_power_$(now()).h5")
    #             mpiexec(cmd -> run(```
    #                 $cmd -n $(nworkers+1) julia --project $kernel 
    #                 $inputfile $outputfile
    #                 --niterations $niterations
    #                 --nwait $nwait
    #                 --ncomponents $ncomponents
    #                 --npartitions $npartitions
    #                 --saveiterates
    #                 ```))        
    #         end
    #     end
    # end

    # run jobs
    # pfraction_all = [1.0, 0.9, 0.5, 0.1]
    # for pfraction in pfraction_all
    #     stepsize = pfraction

    #     # with variance reduction
    #     outputfile = joinpath(directory, "output_$(now()).h5")
    #     mpiexec(cmd -> run(```
    #         $cmd -n $(nworkers+1) julia --project $kernel 
    #         $inputfile $outputfile 
    #         --niterations $niterations 
    #         --ncomponents $ncomponents
    #         --stepsize $stepsize
    #         --pfraction $pfraction
    #         --variancereduced
    #         --saveiterates
    #         ```))

    #     # without variance reduction
    #     outputfile = joinpath(directory, "output_$(now()).h5")
    #     mpiexec(cmd -> run(```
    #         $cmd -n $(nworkers+1) julia --project $kernel 
    #         $inputfile $outputfile 
    #         --niterations $niterations 
    #         --ncomponents $ncomponents
    #         --stepsize $stepsize
    #         --pfraction $pfraction    
    #         --saveiterates
    #         ```))            
    # end

    # nsubpartitions_all = [1, 2, 5, 10]
    # for nsubpartitions in nsubpartitions_all
    #     for stepsize in [1, 1/2, 1/5, 1/10]
    #         # stepsize = 1/nsubpartitions

    #         # with variance reduction
    #         outputfile = joinpath(directory, "output_$(now()).h5")
    #         mpiexec(cmd -> run(```
    #             $cmd -n $(nworkers+1) julia --project $kernel 
    #             $inputfile $outputfile 
    #             --niterations $niterations 
    #             --ncomponents $ncomponents
    #             --stepsize $stepsize
    #             --nsubpartitions $nsubpartitions
    #             --variancereduced
    #             --saveiterates
    #             ```))

    #         # without variance reduction
    #         outputfile = joinpath(directory, "output_$(now()).h5")
    #         mpiexec(cmd -> run(```
    #             $cmd -n $(nworkers+1) julia --project $kernel 
    #             $inputfile $outputfile 
    #             --niterations $niterations 
    #             --ncomponents $ncomponents
    #             --stepsize $stepsize
    #             --nsubpartitions $nsubpartitions
    #             --saveiterates
    #             ```))            
    #     end
    # end

    # compare using nwait and nsubpartitions to control what fraction of the matrix is processed per iteration
    # here, for fractions [1/2, 1/3, 1/4, 1/8]
    # for (nwait, nsubpartitions) in [(12, 2), (6, 1), (4, 1), (12, 3), (3, 1), (12, 4), (6, 2)]
    #     stepsize = 0.9
    #     # for stepsize in [1, 1/2, 1/5, 1/10]        
    #     outputfile = joinpath(directory, "output_$(now()).h5")
    #     mpiexec(cmd -> run(```
    #         $cmd -n $(nworkers+1) julia --project $kernel 
    #         $inputfile $outputfile 
    #         --nwait $nwait
    #         --niterations $niterations
    #         --ncomponents $ncomponents
    #         --stepsize $stepsize
    #         --nsubpartitions $nsubpartitions
    #         --variancereduced
    #         --saveiterates
    #         ```))
    #     # end
    # end

    # niterations = 10
    # stepsize = 0.9
    # for nsubpartitions in [1] # [1, 2, 3, 4, 5]
    #     nwait = 12
    #     for stepsize in [0.1, 0.2, 0.5]
    #         # for nwait in 1:nworkers

    #         # variance-reduced sgd
    #         # outputfile = joinpath(directory, "output_$(now()).h5")
    #         # mpiexec(cmd -> run(```
    #         #     $cmd -n $(nworkers+1) julia --project $kernel 
    #         #     $inputfile $outputfile
    #         #     --inputdataset $inputdataset
    #         #     --nwait $nwait
    #         #     --niterations $niterations
    #         #     --ncomponents $ncomponents
    #         #     --stepsize $stepsize
    #         #     --nsubpartitions $nsubpartitions
    #         #     --variancereduced
    #         #     --saveiterates
    #         #     ```))

    #         # sgd
    #         outputfile = joinpath(directory, "output_$(now()).h5")
    #         mpiexec(cmd -> run(```
    #             $cmd -n $(nworkers+1) julia --project $kernel 
    #             $inputfile $outputfile 
    #             --inputdataset $inputdataset
    #             --nwait $nwait
    #             --niterations $niterations
    #             --ncomponents $ncomponents
    #             --stepsize $stepsize
    #             --nsubpartitions $nsubpartitions
    #             --saveiterates
    #             ```))                
    #     end
    # end

    # # substitute decoding
    # kernel = "./src/pca/power.jl"
    # for npartitions in [6, 8, 10, 11, 12]
    #     code_rate = npartitions / nworkers
    #     for nwait in [npartitions - 1, npartitions]
    #         outputfile = joinpath(directory, "output_power_$(now()).h5")
    #         mpiexec(cmd -> run(```
    #             $cmd -n $(nworkers+1) julia --project $kernel 
    #             $inputfile $outputfile 
    #             --niterations $niterations 
    #             --ncomponents $ncomponents
    #             --saveiterates
    #             --npartitions $npartitions
    #             ```))                        
    #     end
    # end

    # ### compare convergence with and without Diggavi encoding
    # # with Diggavi encoding
    # outputfile = joinpath(directory, "output_$(now()).h5")
    # mpiexec(cmd -> run(```
    #     $cmd -n $(nworkers+1) julia --project $kernel 
    #     $inputfile $outputfile
    #     --inputdataset C
    #     --nwait $(nworkers-1)
    #     --niterations $niterations
    #     --ncomponents $ncomponents
    #     --variancereduced
    #     --saveiterates
    # ```))    
    # # without Diggavi encoding    
    # outputfile = joinpath(directory, "output_$(now()).h5")    
    # mpiexec(cmd -> run(```
    #     $cmd -n $(nworkers+1) julia --project $kernel 
    #     $inputfile $outputfile
    #     --inputdataset X
    #     --nwait $(nworkers-1)
    #     --niterations $niterations
    #     --ncomponents $ncomponents
    #     --variancereduced
    #     --saveiterates
    #     ```))
    
    df = aggregate_benchmark_data(dir=directory, inputfile=inputfile, inputname="X", prefix="output", dfname="df.csv")
end