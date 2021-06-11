# CodedComputing

This repository contains two distributed computing kernels, which we refer to as the latency and PCA kernels, respectively, for collecting latency traces in distributed systems, and associated code for parsing the collected traces. Both kernels are implemented in [Julia](https://julialang.org/), with communication implemented using [MPI](https://www.open-mpi.org/) and [MPIStragglers.jl](https://github.com/severinson/MPIStragglers.jl) (see below for details). 

Both kernels implement the [power iteration](https://en.wikipedia.org/wiki/Power_iteration) method for computing the [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) of a matrix, with multiple worker nodes responsible for computing matrix-matrix products and a coordinator node responsible for aggregating the results. Each iteration consists of

1. the coordinator sending a matrix (the current iterate) to each worker,
2. each workers multiplying the received matrix by a locally stored data matrix, and
3. each worker sending the result back to the coordinator.

Hence, the total latency of each iteration is the sum of the communication and computation latency. For the latency kernel, the computation and overall latency of each worker is recorded separately, whereas for the PCA kernel only the overall iteration latency of each worker is recorded.

## Parsing

Each run of either kernel results in a `.h5` file (i.e., a [HDF5 file](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)), which is parsed to produce a `.csv` file containing the latency associated with each worker and iteration, the explained variance, and the parameters of that run. Each row corresponds to one iteration of the algorithm. Finally, the `.csv` files associated with each run are concatenated to produce a single `.csv` file containing all traces for one experiment setup (i.e., for a particular dataset).

To parse `.h5` files resulting from runs of the PCA kernel, see the `parse_pca_files` and `df_from_output_file` functions in `src/pca/parse.jl`. To parse `.h5` files resulting from runs of the latency kernel, see the corresponding functions in `src/latency/parse.jl`. 

Use the following code to load a `.csv` file into a [DataFrames.jl](https://dataframes.juliadata.org/stable/) DataFrame:
```julia
using CSV, DataFrames
df = DataFrame(CSV.File(<filename>))
```
The rows of this DataFrame correspond to iterations, and the latency of all workers participating in a particular iteration are recorded in the same row. If needed, convert the DataFrame to tall format using the `tall_from_wide` function in `src/Analysis.jl`, which expands the DataFrame such that each row corresponds to one worker for a particular iteration and job.

See the [this repository](https://github.com/severinson/DSAGAnalysis.jl) for further information and analysis of the recorded data.