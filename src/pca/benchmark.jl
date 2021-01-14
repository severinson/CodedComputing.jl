using HDF5, MPI, Dates
using CodedComputing

function write_coded_input(X; inputfile::AbstractString, name="C", coderate=1)
    nc = ceil(Int, size(X, 1) / coderate) # number of coded rows
    C = Matrix(encode_hadamard(X, nc))
    h5write(inputfile, name, C)
    return C
end

function benchmark_main(directory=joinpath("./simulations/1000genomes", "$(now())"))
    
    # MNIST
    # directory = "/home/albin/.julia/dev/CodedComputing/simulations/mnist/2021-01-12T10:34:57.725"

    # 1000 genomes
    # directory = "/home/albin/.julia/dev/CodedComputing/simulations/1000genomes/2021-01-12T17:16:23.558"
    directory = "/home/albin/.julia/dev/CodedComputing/simulations/1000genomes/2021-01-14T13:07:35.092"

    # prepare a directory for storing everything
    mkpath(directory)
    inputfile = joinpath(directory, "inputfile.h5")

    # simulation parameters
    kernel = "./src/pca/pca.jl"
    inputdataset = "X"
    nworkers = 42
    ncomponents = 3
    niterations = 5

    iteratedataset = "V0"
    
    # don't do anything if the input file already exists
    if isfile(inputfile)
        @assert HDF5.ishdf5(inputfile)
    else
        # MNIST dataset
        # X = load_mnist_training()
        # h5write(inputfile, "X", X)

        # 1000 genomes dataset
        X = load_genome_data()
        h5writecsc(inputfile, "X", X)

        V0 = randn(size(X, 2), ncomponents)
        orthogonal!(V0)
        h5write(inputfile, "V0", V0)        
    end

    # Measure iteration time
    stepsize = 1.0
    # nwait = nworkers - 2
    nsubpartitions = 1

    for nreplicas in [2, 3, 1]
        npartitions = div(nworkers, nreplicas) * nsubpartitions
        for nwait in [1, npartitions-2, npartitions-1, npartitions] # 3
            # for nwait in [nworkers-10, nworkers-2, nworkers-1, nworkers]
            # for stepsize in [0.5, 0.9, 1.0]            

            # with variance reduction
            outputfile = joinpath(directory, "output_$(now()).h5")
            mpiexec(cmd -> run(```
                $cmd -n $(nworkers+1) julia --project $kernel 
                $inputfile $outputfile 
                --iteratedataset $iteratedataset
                --niterations $niterations
                --nwait $nwait
                --nreplicas $nreplicas
                --ncomponents $ncomponents
                --stepsize $stepsize
                --nsubpartitions $nsubpartitions
                --variancereduced
                --saveiterates
                ```))            

            # without variance reduction (regular SGD)
            outputfile = joinpath(directory, "output_$(now()).h5")          
            mpiexec(cmd -> run(```
                $cmd -n $(nworkers+1) julia --project $kernel 
                $inputfile $outputfile
                --iteratedataset $iteratedataset        
                --niterations $niterations 
                --nwait $nwait
                --nreplicas $nreplicas
                --ncomponents $ncomponents
                --stepsize $stepsize
                --nsubpartitions $nsubpartitions
                --saveiterates
                ```))            
        end
    end

    df = aggregate_benchmark_data(dir=directory, inputfile=inputfile, inputname="X", prefix="output", dfname="df.csv")
end