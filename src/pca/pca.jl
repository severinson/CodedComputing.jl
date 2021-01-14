using ArgParse, Random

const METADATA_BYTES = 6
const ELEMENT_TYPE = Float64
const CANARY_VALUE = UInt16(2^16 - 1)

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
        "--variancereduced"
            help = "Compute a variance-reduced gradient in each iteration"
            action = :store_true
    end
end

function update_parsed_args!(s::ArgParseSettings, parsed_args)
    parsed_args[:algorithm] = "pca.jl"
    parsed_args
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
    partition_index = ceil(Int, i/nreplicas)
    h5open(inputfile, "r") do fid
        inputdataset in keys(fid) || throw(ArgumentError("$inputdataset is not in $fid"))
        flag, _ = isvalidh5csc(fid, inputdataset)
        if flag
            X = h5readcsc(fid, inputdataset)
            nrows = size(X, 1)
            il = round(Int, (partition_index - 1)/npartitions*nrows + 1)
            iu = round(Int, partition_index/npartitions*nrows)
            return X[il:iu, :]            
        else            
            nrows = size(fid[inputdataset], 1)
            il = round(Int, (partition_index - 1)/npartitions*nrows + 1)
            iu = round(Int, partition_index/npartitions*nrows)
            return fid[inputdataset][il:iu, :]
        end
    end
end

function worker_setup(rank::Integer, nworkers::Integer; ncomponents::Union{Nothing,<:Integer}, kwargs...)
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    isnothing(ncomponents) || 0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))
    localdata = read_localdata(rank, nworkers; kwargs...)
    dims = length(size(localdata))
    dims == 2 || error("Expected localdata to be 2-dimensional, but got data of dimension $dims")
    dimension = size(localdata, 2)

    # default to computing all components
    # TODO: this won't work if an initial iterate is provided but ncomponents isn't set
    if isnothing(ncomponents)
        k = dimension
    else
        k = ncomponents
    end

    recvbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*k)
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*k + METADATA_BYTES)
    localdata, recvbuf, sendbuf
end

function coordinator_setup(nworkers::Integer; inputfile::String, inputdataset::String, iteratedataset, ncomponents, parsed_args...)    
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    isnothing(ncomponents) || 0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))    
    nsamples, dimension = problem_size(inputfile, inputdataset)

    # initial iterate
    if isnothing(iteratedataset) # initialized at random
        k = isnothing(ncomponents) ? dimension : ncomponents
        V = randn(dimension, k)
        orthogonal!(V)
    else # given as an argument and loaded from disk
        h5open(inputfile) do fid
            iteratedataset in keys(fid) || throw(ArgumentError("iterate dataset $iteratedataset not found"))
            V = fid[iteratedataset][:, :]
        end
        ncomponents == size(V, 2) || throw(DimensionMismatch("V has dimensions $(size(V)), but ncomponents is $ncomponents"))
        k = size(V, 2)
    end

    # communication buffers
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*k)
    recvbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*nworkers*k + METADATA_BYTES*nworkers)
    reinterpret(ELEMENT_TYPE, view(sendbuf, :)) .= view(V, :)

    V, recvbuf, sendbuf
end

function worker_task!(recvbuf, sendbuf, localdata; state=nothing, pfraction::Real, nsubpartitions::Integer, ncomponents, kwargs...)
    0 < pfraction <= 1 || throw(DomainError(pfraction, "pfraction must be in (0, 1]"))
    isnothing(ncomponents) || 0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))        
    sizeof(recvbuf) + METADATA_BYTES == sizeof(sendbuf) || throw(DimensionMismatch("recvbuf has size $(sizeof(recvbuf)), but sendbuf has size $(sizeof(sendbuf))"))
    dimension = size(localdata, 2)
    1 <= nsubpartitions <= dimension || throw(DimensionMismatch("nsubpartitions is $nsubpartitions, but the dimension is $dimension"))
    k = isnothing(ncomponents) ? div(length(reinterpret(ELEMENT_TYPE, recvbuf)), dimension) : ncomponents

    # format the recvbuf into a matrix we can operate on
    length(reinterpret(ELEMENT_TYPE, recvbuf)) == dimension*k || throw(DimensionMismatch("recvbuf has length $(length(reinterpret(ELEMENT_TYPE, recvbuf))), but the data dimension is $dimension and ncomponents is $k"))
    V = reshape(reinterpret(ELEMENT_TYPE, recvbuf), dimension, k)
    Xw = localdata    

    # initialize state
    if isnothing(state)
        max_rows = ceil(Int, ceil(size(Xw, 1)/nsubpartitions) * pfraction)
        W = Matrix{eltype(V)}(undef, max_rows, size(V, 2))
    else
        W::Matrix{eltype(V)} = state
    end

    # select a sub-partition at random
    subpartition_index = rand(1:nsubpartitions)
    il = round(Int, (subpartition_index - 1)/nsubpartitions*size(Xw, 1) + 1)
    iu = round(Int, subpartition_index/nsubpartitions*size(Xw, 1))

    # select a fraction pfraction of that partition at random
    p = shuffle!(collect(il:iu))
    j = round(Int, pfraction*length(p))
    j = max(1, j)
    Xwv = view(Xw, view(p, 1:j), :)

    # do the computation
    Wv = view(W, 1:size(Xwv, 1), :)
    mul!(Wv, Xwv, V)
    mul!(V, Xwv', Wv)
    
    # populate the send buffer
    metadata = reinterpret(UInt16, view(sendbuf, 1:METADATA_BYTES))
    metadata[1] = CANARY_VALUE
    metadata[2] = rank
    metadata[3] = subpartition_index
    @views sendbuf[METADATA_BYTES+1:end] .= recvbuf[:]
    W
end

data_view(recvbuf) = reinterpret(ELEMENT_TYPE, @view recvbuf[METADATA_BYTES+1:end])
metadata_view(recvbuf) = view(recvbuf, 1:METADATA_BYTES)

function update_gradient_sgd!(∇, recvbufs, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, nreplicas, pfraction, nsubpartitions, kwargs...)
    length(recvbufs) == length(repochs) || throw(DimensionMismatch("recvbufs has dimension $(length(recvbufs)), but repochs has dimension $(length(repochs))"))
    0 < pfraction <= 1 || throw(DomainError(pfraction, "pfraction must be in (0, 1]"))
    0 < nreplicas || throw(DomainError(nreplicas, "nreplicas must be positive"))
    epoch <= 1 || !isnothing(state) || error("expected state to be initiated for epoch > 1")
    nworkers = length(recvbufs)
    mod(nworkers, nreplicas) == 0 || throw(ArgumentError("nworkers must be divisible by nreplicas"))
    npartitions = div(nworkers, nreplicas) * nsubpartitions

    # record the epoch at which each partition was last updated
    if isnothing(state)
        uepochs = zeros(Int, npartitions)        
    else
        uepochs = state
    end

    # add at most 1 replica of each partition to the overall gradient
    # the partitions are arranged sequentially, so if there are 2 partitions and 3 replicas, then
    # Vs is of length 6, and its elements correspond to partitions [1, 1, 1, 2, 2, 2]
    ∇ .= 0
    nresults = 0
    for worker_index in 1:nworkers

        # skip workers that we've never received anything from
        if repochs[worker_index] == 0
            continue
        end

        metadata = reinterpret(UInt16, metadata_view(recvbufs[worker_index]))
        if length(metadata) != 3
            @error "received incorrectly formatted metadata from the $(worker_index)-th worker in epoch $epoch: $metadata"
            continue
        end
        canary, worker_rank, subpartition_index = metadata
        if  canary != CANARY_VALUE
            @error "recieved incorrect canary value from the $(worker_index)-th worker in epoch $epoch: $canary"
            continue
        end
        if worker_rank != worker_index
            @error "unexpected rank for the $(worker_index)-th worker in epoch $epoch: $worker_rank"
            continue
        end
        if !(0 < subpartition_index <= nsubpartitions)
            @error "received incorrect sub-partition index from the $(worker_index)-th worker in epoch $epoch: $subpartition_index "
            continue
        end
        replica_index = ceil(Int, worker_index/nreplicas)
        partition_index = (replica_index-1)*nsubpartitions + subpartition_index

        # don't do anything if we didn't receive from this worker this epoch,
        # or if we've already updated that partition this epoch
        if repochs[worker_index] < epoch || uepochs[partition_index] == epoch
            continue
        end

        # add the sub-gradient computed by this worker
        uepochs[partition_index] = epoch
        Vw = reshape(data_view(recvbufs[worker_index]), size(∇)...)
        ∇ .-= Vw
        nresults += 1
    end

    # scale the (stochastic) gradient to make it unbiased estimate of the true gradient
    ∇ .*= (npartitions / nresults) / pfraction
    uepochs
end

function update_gradient_vr!(∇, recvbufs, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, nreplicas::Integer, pfraction::Real, nsubpartitions::Integer, kwargs...)
    length(recvbufs) == length(repochs) || throw(DimensionMismatch("recvbufs has dimension $(length(recvbufs)), but repochs has dimension $(length(repochs))"))
    0 < pfraction <= 1 || throw(DomainError(pfraction, "pfraction must be in (0, 1]"))
    0 < nreplicas || throw(DomainError(nreplicas, "nreplicas must be positive"))
    epoch <= 1 || !isnothing(state) || error("expected state to be initiated for epoch > 1")
    nworkers = length(recvbufs)
    mod(nworkers, nreplicas) == 0 || throw(ArgumentError("nworkers must be divisible by nreplicas"))
    npartitions = div(nworkers, nreplicas) * nsubpartitions

    # record the epoch at which each partition was last updated
    # store the previously computed partial gradients
    if isnothing(state)
        uepochs = zeros(Int, npartitions)
        ∇s = [zeros(eltype(∇), size(∇)...) for _ in 1:npartitions]
    else
        uepochs, ∇s = state
    end    

    # iterate over the received partial gradients
    # cache any received partial gradient that is newer than what we currently store
    for worker_index in 1:nworkers

        # skip workers that we've never received anything from
        if repochs[worker_index] == 0
            continue
        end

        metadata = reinterpret(UInt16, metadata_view(recvbufs[worker_index]))
        if length(metadata) != 3
            @error "received incorrectly formatted metadata from the $(worker_index)-th worker in epoch $epoch: $metadata"
            continue
        end
        canary, worker_rank, subpartition_index = metadata
        if  canary != CANARY_VALUE
            @error "recieved incorrect canary value from the $(worker_index)-th worker in epoch $epoch: $canary"
            continue
        end
        if worker_rank != worker_index
            @error "unexpected rank for the $(worker_index)-th worker in epoch $epoch: $worker_rank"
            continue
        end
        if !(0 < subpartition_index <= nsubpartitions)
            @error "received incorrect sub-partition index from the $(worker_index)-th worker in epoch $epoch: $subpartition_index "
            continue
        end
        replica_index = ceil(Int, worker_index/nreplicas)
        partition_index = (replica_index-1)*nsubpartitions + subpartition_index

        # skip updates that contain no new information
        if repochs[worker_index] <= uepochs[partition_index]
            continue
        end

        # store the received partial gradient
        ∇w = reshape(data_view(recvbufs[worker_index]), size(∇)...)
        ∇s[partition_index] .= ∇w
        uepochs[partition_index] = repochs[worker_index]
    end

    # estimate the gradient by the sum of the cached partial gradients
    ∇ .= 0
    for ∇i in ∇s
        ∇ .-= ∇i
    end

    # scale the gradient by the number of non-zero partial gradients,
    # important for the first few iterations when some entries of ∇s may be zero
    s = npartitions / (npartitions - sum(iszero, uepochs))
    if !isone(s) && !isinf(s)
        ∇ .*= s
    end

    uepochs, ∇s
end

update_gradient!(args...; variancereduced::Bool, kwargs...) = variancereduced ? update_gradient_vr!(args...; kwargs...) : update_gradient_sgd!(args...; kwargs...)

function update_iterate!(V, ∇; state=nothing, stepsize, kwargs...)
    size(V) == size(∇) || throw(DimensionMismatch("V has dimensions $(size(B)), but ∇ has dimensions $(size(∇))"))
    for I in CartesianIndices(V)
        V[I] -= stepsize * (∇[I] + V[I])
    end
    orthogonal!(V)
    state
end

function coordinator_task!(V, ∇, recvbufs, sendbuf, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, kwargs...)
    isnothing(state) || length(state) == 2 || throw(ArgumentError("expected state to be nothing or a tuple of length 2, but got $state"))
    if isnothing(state)
        gradient_state = update_gradient!(∇, recvbufs, epoch, repochs; kwargs...)
        iterate_state = update_iterate!(V, ∇; kwargs...)
    else
        gradient_state, iterate_state = state
        gradient_state = update_gradient!(∇, recvbufs, epoch, repochs; state=gradient_state, kwargs...)
        iterate_state = update_iterate!(V, ∇; state=iterate_state, kwargs...)
    end
    reinterpret(ELEMENT_TYPE, view(sendbuf, :)) .= view(V, :)        
    gradient_state, iterate_state
end

include("common.jl")