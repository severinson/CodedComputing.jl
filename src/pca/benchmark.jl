using HDF5, MPI, Dates
using CodedComputing

function write_coded_input(X; inputfile::AbstractString, name="C", coderate=1)
    nc = ceil(Int, size(X, 1) / coderate) # number of coded rows
    C = Matrix(encode_hadamard(X, nc))
    h5write(inputfile, name, C)
    return C
end

function benchmark_main(directory=joinpath("./simulations/1000genomes", "$(now())"))
    
    # prepare a directory for storing everything
    mkpath(directory)
    inputfile = joinpath(directory, "inputfile.h5")
    
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
    end

    # simulation parameters
    kernel = "./src/pca/pca.jl"
    inputdataset = "X"
    nworkers = 10
    ncomponents = 3
    niterations = 10

    # Measure iteration time
    stepsize = 0.9
    nwait = nworkers - 1
    nsubpartitions = 1
    
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

    df = aggregate_benchmark_data(dir=directory, inputfile=inputfile, inputname="X", prefix="output", dfname="df.csv")
end