using ArgParse, Random

const METADATA_BYTES = 2
const ELEMENT_TYPE = Float64

function update_argsettings!(s::ArgParseSettings)
    @add_arg_table s begin
        "--pfraction"
            help = "Fraction of the data stored at each worker that should be used to compute the gradient"
            arg_type = Float64
            default = 1.0
            range_tester = (x) -> 0 < x <= 1
        "--nsubpartitions"
            help = "Number of sub-partitions to split the data stored at each worker into"
            arg_type = Int
            default = 1
            range_tester = (x) -> x >= 1
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

    recvbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*k)
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*k + METADATA_BYTES)
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
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*k)
    recvbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*nworkers*k + METADATA_BYTES*nworkers)

    # iterate, initialized at random
    V = randn(dimension, k)
    orthogonal!(V)
    reinterpret(ELEMENT_TYPE, view(sendbuf, :)) .= view(V, :)

    V, recvbuf, sendbuf
end

function worker_task!(recvbuf, sendbuf, localdata; state=nothing, pfraction::Real=1, nsubpartitions::Integer, ncomponents, kwargs...)
    0 < pfraction <= 1 || throw(DomainError(pfraction, "pfraction must be in (0, 1]"))
    isnothing(ncomponents) || 0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))        
    sizeof(recvbuf) + METADATA_BYTES == sizeof(sendbuf) || throw(DimensionMismatch("recvbuf has size $(sizeof(recvbuf)), but sendbuf has size $(sizeof(sendbuf))"))
    dimension = size(localdata, 2)
    1 <= nsubpartitions <= dimension || throw(DimensionMismatch("nsubpartitions is $nsubpartitions, but the dimension is $dimension"))

    # default to computing all components
    if isnothing(ncomponents)
        k = dimension
    else
        k = ncomponents
    end        

    # format the recvbuf into a matrix we can operate on
    length(reinterpret(ELEMENT_TYPE, recvbuf)) == dimension*k || throw(DimensionMismatch("recvbuf has length $(length(reinterpret(ELEMENT_TYPE, recvbuf))), but the data dimension is $dimension and ncomponents is $k"))
    V = reshape(reinterpret(ELEMENT_TYPE, recvbuf), dimension, k)
    Xw = localdata

    # initialize state
    if isnothing(state)
        max_rows = ceil(Int, ceil(size(Xw, 1)/nsubpartitions) * pfraction)
        W = Matrix{eltype(V)}(undef, max_rows, size(V, 2))
    else
        W = state
    end

    # select a sub-partition at random
    i = rand(1:nsubpartitions)
    il = round(Int, (i - 1)/nsubpartitions*size(Xw, 1) + 1)
    iu = round(Int, i/nsubpartitions*size(Xw, 1))

    # select a fraction pfraction of that partition at random
    p = shuffle!(collect(il:iu))
    j = round(Int, pfraction*length(p))
    j = max(1, j)
    Xwv = view(Xw, view(p, 1:j), :)
    Wv = view(W, 1:size(Xwv, 1), :)

    # do the computation
    mul!(Wv, Xwv, V)
    mul!(V, Xwv', Wv)

    @views sendbuf[METADATA_BYTES+1:end] .= recvbuf[:]
    W
end

data_view(recvbuf) = reinterpret(ELEMENT_TYPE, @view recvbuf[METADATA_BYTES+1:end])
metadata_view(recvbuf) = view(recvbuf, 1:METADATA_BYTES)

function update_gradient!(∇, recvbufs, sendbuf, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, nreplicas=1, pfraction=1, nsubpartitions, kwargs...)
    length(recvbufs) == length(repochs) || throw(DimensionMismatch("recvbufs has dimension $(length(recvbufs)), but repochs has dimension $(length(repochs))"))
    0 < pfraction <= 1 || throw(DomainError(pfraction, "pfraction must be in (0, 1]"))
    0 < nreplicas || throw(DomainError(nreplicas, "nreplicas must be positive"))
    nworkers = length(recvbufs)
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
                Vi = reshape(data_view(recvbufs[i]), size(∇)...)
                ∇ .+= Vi
                nresults += 1
                break
            end
        end
    end

    # scale the (stochastic) gradient to make it unbiased estimate of the true gradient
    ∇ .*= (nworkers / nresults) / pfraction * nsubpartitions

    state
end

function update_iterate!(V, ∇, sendbuf, epoch, repochs; state=nothing, stepsize=1, kwargs...)
    size(V) == size(∇) || throw(DimensionMismatch("V has dimensions $(size(B)), but ∇ has dimensions $(size(∇))"))
    for I in CartesianIndices(V)
        V[I] -= stepsize * (∇[I] + V[I])
    end
    orthogonal!(V)
    reinterpret(ELEMENT_TYPE, view(sendbuf, :)) .= view(V, :)
    state
end

include("common.jl")