using HDF5, MPI, Dates
using CodedComputing

function test_matrix(n, m, k)
    G = randn(n, m)
    P = orthogonal!(randn(m, k))
    X = (G*P)*P'
    X .+= randn(size(G)...)./100
    X
end

function benchmark_main(n=10000, m=5000, k=100, directory=joinpath("./simulations", "$(now())"))
    
    # prepare a directory for storing everything
    inputfile = joinpath(directory, "inputfile.h5")
    mkpath(directory)

    # generate a random input data matrix if it doesn't already exist
    if !isfile(inputfile)
        X = test_matrix(n, m, k)
        h5write(inputfile, "X", X)
    end
    @assert HDF5.ishdf5(inputfile)

    # simulation parameters
    kernel = "./src/pca/pca.jl"
    nworkers = 12
    ncomponents = k
    niterations = 20

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

    niterations = 10
    stepsize = 0.9
    for nsubpartitions in [1, 2, 3, 4, 5]
        for nwait in 1:nworkers

            # variance-reduced sgd
            outputfile = joinpath(directory, "output_$(now()).h5")
            mpiexec(cmd -> run(```
                $cmd -n $(nworkers+1) julia --project $kernel 
                $inputfile $outputfile 
                --nwait $nwait
                --niterations $niterations
                --ncomponents $ncomponents
                --stepsize $stepsize
                --nsubpartitions $nsubpartitions
                --variancereduced
                --saveiterates
                ```))

            # sgd
            outputfile = joinpath(directory, "output_$(now()).h5")
            mpiexec(cmd -> run(```
                $cmd -n $(nworkers+1) julia --project $kernel 
                $inputfile $outputfile 
                --nwait $nwait
                --niterations $niterations
                --ncomponents $ncomponents
                --stepsize $stepsize
                --nsubpartitions $nsubpartitions
                --variancereduced
                --saveiterates
                ```))                
        end
    end

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

    df = aggregate_benchmark_data(dir=directory, inputfile=inputfile, inputname="X", prefix="output", dfname="df.csv")
end