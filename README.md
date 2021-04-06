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

See the bottom of this file for a description of the columns of the resulting DataFrame.

## Amazon Web Services traces

We have collected latency traces on Amazon Web Services (AWS) using these two kernels, which are available as `.csv` files [here](https://www.dropbox.com/sh/wa3s4yeasqeko5h/AABLPknDQO6TU2s-NDhzpI1Ia?dl=0). Traces collected using the latency and PCA kernels are prefixed with `latency` and `pca`, respectively. For the PCA files, the next section of the filename indicates the dataset used (see below). Finally, the last two sections of the filename indicates the AWS [instance type](https://aws.amazon.com/ec2/instance-types/) and [region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-available-regions) used.

The `.h5` files and corresponding `.csv` files are available in the corresponding `.zip` files in the traces directory.

<!-- 
We give a more detailed overview of both kernels below.

**Latency kernel**

1. Before the start of the computation, each worker generates a sparse matrix, denoted by `X`, using the `sprand` function, and a dense matrix, denoted by `W`, using the `randn` function. The dimensions of `X` and `W` and the density of `X` are given as command-line arguments. Both matrices contain `Float64` entries.
2. At the start of each iteration, the coordinator sends 


Files suffixed with `tall` are converted from wide to tall format. Each row corresponds to the latency of an individual worker. 

I want to update the kernels to use H5Sparse

- Kernels and associated files
- Code for creating the datasets
- Code for parsing the .h5 files
- Code for parsing the .csv files

-->


### Datasets

For the PCA kernel, we have collected traces using two different datasets, denoted by `1000enomes` and `sprand`, respectively. The `1000genomes` matrix is derived from the [1000 Genomes dataset](https://www.internationalgenome.org/). More precisely, we consider a binary representation of the data, where a non-zero entry in the `(i, j)`-th position indicates that the genome of the `i`-th subject differs from that of the reference genome in the `j`-th position, i.e., rows correspond to persons and columns to positions in the genome. The matrix is created by concatenating the sub-matrices corresponding to each chromosome and then randomly permuting the columns, which we do to break up dense blocks. It is a sparse matrix with dimensions `(2504, 81 271 767)` and density about 5.36%. When stored in compressed column format, with 64-bit indices, the matrix is about 100 gigabytes. The `sprand` matrix is created in Julia with the command `sprand(2504, 3600000, 0.05360388070027386)`.

Both matrices are available [here](https://www.dropbox.com/sh/ak5d9elhra2h4in/AAB2qqleIxAYTpVlxHba_q_0a?dl=0) and can be read with the [H5Sparse.jl](https://github.com/severinson/H5Sparse.jl) Julia package.



### Configuration

All experiments are carried out using the configuration listed below.

```julia
# setup
pkg> add MPI, MKL, MKLSparse
> ENV["JULIA_MPI_BINARY"] = "system"
pkg> build MPI
pkg> build MKL

# package versions
pkg> status MPI
[da04e1cc] MPI v0.16.1
pkg> status MKL
[33e6dc65] MKL v0.4.0
pkg> status MKLSparse
[0c723cd3] MKLSparse v1.1.0

# versioninfo
> versioninfo()
Julia Version 1.5.4
Commit 69fcb5745b (2021-03-11 19:13 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Platinum 8124M CPU @ 3.00GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-9.0.1 (ORCJIT, skylake-avx512)  

> MPI.Get_library_version()
"Open MPI v4.0.3, package: Open MPI root@b0fe8a010177 Distribution, ident: 4.0.3, repo rev: v4.0.3, Mar 03, 2020"

> MPI.Get_version()
v"3.1.0"  

> LinearAlgebra.versioninfo()
BLAS: libmkl_rt.so.1
LAPACK: libmkl_rt.so.1

> BLAS.vendor()
:mkl
```

## DataFrame columns

Each row of the concatenated DataFrame corresponds to one iteration of a particular job, and the DataFrame has columns:

* `iteration`: Iteration index
* `jobid`: Unique ID for each run of the kernel, i.e., each unique `jobid` corresponds to one `.h5` file
* `latency`: Overall latency of the iteration
* `latency_worker_<i>`: Latency of the `i`-th worker in this iteration
* `mse`: Explained variance of the iterate computed in this iteration
* `nbytes`: Total number of bytes transferred in each direction per iteration
* `ncolumns`: Number of columns of the data matrix
* `ncomponents`: Number of PCs computed
* `niterations`: Total number of iterations of this job
* `nostale`: If `true`, stale results received by the coordinator are discarded (only relevant when `variancereduced` is `true`)
* `npartitions`: Total number of data partitions
* `nreplicas`: Number of replicas of each partition of the data matrix
* `nrows`: Number of rows of the data matrix
* `nsubpartitions`: Number of sub-partitions per worker
* `nwait`: Number of workers waited for in each iterations
* `nworkers`: Total number of workers for this run
* `repoch_worker_<i>`: Iteration that the result received from the `i`-th worker was computed for, i.e., the result is stale if it is less than `iteration`
* `saveiterates`: If `true`, iterates were saved for this job
* `stepsize`: Step size used for the gradient update
* `time`: Cumulative iteration latency up this iteration
* `update_latency`: Latency associated with computing the updated iterate at the coordinator
* `variancereduced`: If `true`, the variance-reduced DSAG method was used for this job, whereas, if `false`, SGD was used
* `worker_flops`: Estimated number of FLOPS per worker and iteration