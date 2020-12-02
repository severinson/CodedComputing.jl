using ArgParse, Random

function update_argsettings!(s::ArgParseSettings)
    @add_arg_table s begin
        "--pfraction"
            help = "Fraction of the data stored at each worker that should be used to compute the gradient"
            arg_type = Float64
            default = 1.0
            range_tester = (x) -> 0 < x <= 1
        "--stepsize"
            help = "Gradient descent step size"
            arg_type = Float64
            default = 1.0
            range_tester = (x) -> x > 0
    end    
end

function update_parsed_args!(s::ArgParseSettings, parsed_args)
    parsed_args[:algorithm] = "pca.jl"
end

function problem_size(filename::String, dataset::String)
    HDF5.ishdf5(filename) || throw(ArgumentError("$filename isn't an HDF5 file"))
    h5open(filename, "r") do fid
        dataset in keys(fid) || throw(ArgumentError("$dataset is not in $fid"))
        flag, _ = isvalidh5csc(fid, dataset)
        if flag
            g = fid[dataset]
            return g["m"][], g["n"][]
        end
        return size(fid[dataset])
    end
end

function read_localdata(i::Integer, nworkers::Integer; inputfile::String, inputdataset::String, nreplicas::Integer, kwargs...)
    HDF5.ishdf5(inputfile) || throw(ArgumentError("$inputfile isn't an HDF5 file"))
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    0 < nreplicas || throw(DomainError(nreplicas, "nreplicas must be positive"))
    0 < i <= nworkers || throw(DomainError(i, "i must be in [1, nworkers]"))
    mod(nworkers, nreplicas) == 0 || throw(ArgumentError("nworkers must be divisible by nreplicas"))
    npartitions = div(nworkers, nreplicas)
    h5open(inputfile, "r") do fid
        inputdataset in keys(fid) || throw(ArgumentError("$inputdataset is not in $fid"))
        flag, _ = isvalidh5csc(fid, inputdataset)
        if flag
            X = h5readcsc(fid, inputdataset)
            m = size(X, 1)
            il = round(Int, (i - 1)/npartitions*m + 1)
            iu = round(Int, i/npartitions*m)
            return X[il:iu, :]            
        else            
            n, m = size(fid[inputdataset])
            il = round(Int, (i - 1)/npartitions*n + 1)
            iu = round(Int, i/npartitions*n)
            return fid[inputdataset][il:iu, :]
        end
    end
end

function worker_setup(rank::Integer, nworkers::Integer; ncomponents, kwargs...)
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    isnothing(ncomponents) || 0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))
    localdata = read_localdata(rank, nworkers; kwargs...)
    dims = length(size(localdata))
    dims == 2 || error("Expected localdata to be 2-dimensional, but got data of dimension $dims")
    dimension = size(localdata, 2)

    # default to computing all components
    if isnothing(ncomponents)
        k = dimension
    else
        k = ncomponents
    end

    recvbuf = Matrix{Float64}(undef, dimension, k)
    sendbuf = Matrix{Float64}(undef, dimension, k)
    localdata, recvbuf, sendbuf
end

function coordinator_setup(nworkers::Integer; inputfile::String, inputdataset::String, ncomponents, parsed_args...)    
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    isnothing(ncomponents) || 0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))
    nsamples, dimension = problem_size(inputfile, inputdataset)

    # default to computing all components
    if isnothing(ncomponents)
        k = dimension
    else
        k = ncomponents
    end    

    # communication buffers
    sendbuf = Matrix{Float64}(undef, dimension, k)
    recvbuf = Matrix{Float64}(undef, dimension, nworkers*k)

    # iterate, initialized at random
    V = randn(dimension, k)
    orthogonal!(V)
    view(sendbuf, :) .= view(V, :)

    V, recvbuf, sendbuf
end

function worker_task!(Vrecv, Vsend, localdata; state=nothing, pfraction=1, kwargs...)
    V = Vrecv
    Xw = localdata
    0 < pfraction <= 1 || throw(DomainError(pfraction, "pfraction must be in (0, 1]"))
    if isnothing(state)
        W = Matrix{eltype(V)}(undef, size(Xw, 1), size(V, 2))
        p = collect(1:size(Xw, 1))
    else
        W, p = state
    end

    # select a fraction pfraction of the locally stored rows at random
    shuffle!(p)
    i = round(Int, pfraction*size(Xw, 1))
    i = max(1, i)
    Xwv = view(Xw, view(p, 1:i), :)
    Wv = view(W, 1:size(Xwv, 1), :)

    # do the computation
    mul!(Wv, Xwv, V)
    mul!(V, Xwv', Wv)
    Vsend .= Vrecv

    W, p
end

function update_gradient!(∇, recvbufs, sendbuf, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, nreplicas=1, pfraction=1, kwargs...)
    Vs = [reshape(buf, size(∇)...) for buf in recvbufs]
    length(Vs) == length(repochs) || throw(DimensionMismatch("Vs has dimension $(length(Vs)), but repochs has dimension $(length(repochs))"))
    0 < pfraction <= 1 || throw(DomainError(pfraction, "pfraction must be in (0, 1]"))
    0 < nreplicas || throw(DomainError(nreplicas, "nreplicas must be positive"))
    nworkers = length(Vs)    
    mod(nworkers, nreplicas) == 0 || throw(ArgumentError("nworkers must be divisible by nreplicas"))
    npartitions = div(nworkers, nreplicas)
    ∇ .= 0
    nresults = 0

    # add at most 1 replica of each partition to the overall gradient
    # the partitions are arranged sequentially, so if there are 2 partitions and 3 replicas, then
    # Vs is of length 6, and its elements correspond to partitions [1, 1, 1, 2, 2, 2]
    for partition in 1:npartitions
        for replica in 1:nreplicas
            i = (partition-1)*nreplicas + replica
            if repochs[i] == epoch
                ∇ .+= Vs[i]
                nresults += 1
                break
            end
        end
    end

    # scale the (stochastic) gradient to make it unbiased estimate of the true gradient
    ∇ .*= nworkers / nresults / pfraction

    state
end

function update_iterate!(V, ∇, sendbuf, epoch, repochs; state=nothing, stepsize=1, kwargs...)
    size(V) == size(∇) || throw(DimensionMismatch("V has dimensions $(size(B)), but ∇ has dimensions $(size(∇))"))
    for I in CartesianIndices(V)
        V[I] -= stepsize * (∇[I] + V[I])
    end
    orthogonal!(V)
    view(sendbuf, :) .= view(V, :)
    state
end

include("common.jl")