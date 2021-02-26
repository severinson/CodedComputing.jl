using ArgParse, Random

const METADATA_BYTES = 6
const ELEMENT_TYPE = Float64
const CANARY_VALUE = UInt16(2^16 - 1)

function update_argsettings!(s::ArgParseSettings)
    @add_arg_table s begin
        "--ncomponents"
            help = "Number of principal components to compute"
            required = true
            arg_type = Int            
            range_tester = (x) -> x >= 1        
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
        "--nostale"
            help = "If set, do not store stale gradients (to conform with SAG)"
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
    dimension, nsamples = problem_size(inputfile, inputdataset)    
    h5open(inputfile, "r") do fid
        inputdataset in keys(fid) || throw(ArgumentError("$inputdataset is not in $fid"))
        flag, _ = isvalidh5csc(fid, inputdataset)
        if flag
            il = round(Int, (partition_index - 1)/npartitions*nsamples + 1)
            iu = round(Int, partition_index/npartitions*nsamples)
            return h5readcsc(fid, inputdataset, il, iu)
        else            
            il = round(Int, (partition_index - 1)/npartitions*nsamples + 1)
            iu = round(Int, partition_index/npartitions*nsamples)
            return fid[inputdataset][:, il:iu]
        end
    end
end

function worker_setup(rank::Integer, nworkers::Integer; ncomponents::Integer, kwargs...)
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    isnothing(ncomponents) || 0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))
    localdata = read_localdata(rank, nworkers; kwargs...)
    dims = length(size(localdata))
    dims == 2 || error("Expected localdata to be 2-dimensional, but got data of dimension $dims")
    dimension = size(localdata, 1)
    recvbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*ncomponents)
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*ncomponents + METADATA_BYTES)
    localdata, recvbuf, sendbuf
end

function coordinator_setup(nworkers::Integer; inputfile::String, inputdataset::String, iteratedataset, ncomponents::Integer, parsed_args...)    
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    isnothing(ncomponents) || 0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))    
    dimension, nsamples = problem_size(inputfile, inputdataset)

    # initial iterate
    if isnothing(iteratedataset) # initialized at random
        V = randn(dimension, ncomponents)
        orthogonal!(V)
    else # given as an argument and loaded from disk
        h5open(inputfile) do fid
            iteratedataset in keys(fid) || throw(ArgumentError("iterate dataset $iteratedataset not found"))
            V = fid[iteratedataset][:, :]
        end
        ncomponents == size(V, 2) || throw(DimensionMismatch("V has dimensions $(size(V)), but ncomponents is $ncomponents"))
    end

    # communication buffers
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*ncomponents)
    recvbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*nworkers*ncomponents + METADATA_BYTES*nworkers)
    reinterpret(ELEMENT_TYPE, view(sendbuf, :)) .= view(V, :)

    V, recvbuf, sendbuf
end

function worker_task!(recvbuf, sendbuf, localdata; state=nothing, nsubpartitions::Integer, ncomponents::Integer, kwargs...)
    0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))        
    sizeof(recvbuf) + METADATA_BYTES == sizeof(sendbuf) || throw(DimensionMismatch("recvbuf has size $(sizeof(recvbuf)), but sendbuf has size $(sizeof(sendbuf))"))
    dimension, nlocalsamples = size(localdata)
    1 <= nsubpartitions <= dimension || throw(DimensionMismatch("nsubpartitions is $nsubpartitions, but the dimension is $dimension"))

    # format the recvbuf into a matrix we can operate on
    length(reinterpret(ELEMENT_TYPE, recvbuf)) == dimension*ncomponents || throw(DimensionMismatch("recvbuf has length $(length(reinterpret(ELEMENT_TYPE, recvbuf))), but the data dimension is $dimension and ncomponents is $k"))
    V = reshape(reinterpret(ELEMENT_TYPE, recvbuf), dimension, ncomponents)
    Xw = localdata

    # prepare working memory
    if isnothing(state) # first iteration
        max_samples = ceil(Int, nlocalsamples/nsubpartitions) # max number of samples processed per iteration
        W = Matrix{eltype(V)}(undef, max_samples, ncomponents)
    else # subsequent iterations
        W::Matrix{eltype(V)} = state
    end

    # select a sub-partition at random
    subpartition_index = rand(1:nsubpartitions)
    il = round(Int, (subpartition_index - 1)/nsubpartitions*nlocalsamples + 1)
    iu = round(Int, subpartition_index/nsubpartitions*nlocalsamples)
    Xwv = view(Xw, :, il:iu)

    # do the computation
    Wv = view(W, 1:size(Xwv, 2), :)
    mul!(Wv, Xwv', V)
    mul!(V, Xwv, Wv)
    
    # populate the send buffer
    metadata = reinterpret(UInt16, view(sendbuf, 1:METADATA_BYTES))
    metadata[1] = CANARY_VALUE
    metadata[2] = rank
    metadata[3] = subpartition_index
    @views sendbuf[METADATA_BYTES+1:end] .= recvbuf[:] # V is aliased to recvbuf
    W
end

data_view(recvbuf) = reinterpret(ELEMENT_TYPE, @view recvbuf[METADATA_BYTES+1:end])
metadata_view(recvbuf) = view(recvbuf, 1:METADATA_BYTES)

function update_gradient_sgd!(∇, recvbufs, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, nreplicas, nsubpartitions, kwargs...)
    length(recvbufs) == length(repochs) || throw(DimensionMismatch("recvbufs has dimension $(length(recvbufs)), but repochs has dimension $(length(repochs))"))
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
    ∇ .*= npartitions / nresults
    uepochs
end

function update_gradient_vr!(∇, recvbufs, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, nostale::Bool, nreplicas::Integer, nsubpartitions::Integer, kwargs...)
    length(recvbufs) == length(repochs) || throw(DimensionMismatch("recvbufs has dimension $(length(recvbufs)), but repochs has dimension $(length(repochs))"))
    0 < nreplicas || throw(DomainError(nreplicas, "nreplicas must be positive"))
    epoch <= 1 || !isnothing(state) || error("expected state to be initiated for epoch > 1")
    nworkers = length(recvbufs)
    mod(nworkers, nreplicas) == 0 || throw(ArgumentError("nworkers must be divisible by nreplicas"))
    npartitions = div(nworkers, nreplicas) * nsubpartitions

    # record the epoch at which each partition was last updated
    # store the previously computed partial gradients
    if isnothing(state)
        uepochs = zeros(Int, npartitions)
        partition_worker_map = zeros(Int, npartitions)
        G = copy(∇)
        ∇s = [zeros(eltype(∇), size(∇)...) for _ in 1:npartitions]
    else
        uepochs, partition_worker_map, G, ∇s = state
        partition_worker_map .= 0
    end    

    # iterate over the received partial gradients to record which of them are newer than what we currently have
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

        # discard stale gradients if the nostale option is set
        if nostale && repochs[worker_index] != epoch
            continue
        end

        # skip updates that contain no new information
        if repochs[worker_index] <= uepochs[partition_index]
            continue
        end

        # record which worker has updates for which partition and how new it is
        partition_worker_map[partition_index] = worker_index
        uepochs[partition_index] = repochs[worker_index]        
    end

    # number of new subgradients received in this iteration
    nupdated = sum(!iszero, partition_worker_map)

    # iterate over the partitions to update the locally stored partial gradients
    for partition_index in 1:npartitions
        worker_index = partition_worker_map[partition_index]
        if iszero(worker_index) # zero indicates no update for this partition
            continue
        end        
        ∇w = reshape(data_view(recvbufs[worker_index]), size(∇)...)

        # there are two options for computing the new gradient sum
        # 1. set G to 0 and compute the sum anew over all ∇s
        # 2. for each new subgradient, subtract the previously stored subgradient and add the new one
        # option 1 requires npartitions+1 operations and option 2 requires two operations per updated subgradient
        # this implementation picks the option requiring fewer operations
        if 2*nupdated < npartitions + 1
            # notice the flipped signs (has to do with how the worker task is implemented)
            G .+= ∇s[partition_index] # remove the previous subgradient
            G .-= ∇w # add the new subgradient
        end
        ∇s[partition_index] .= ∇w # store the new subgradient    
    end

    # option 1 gradient update (if we didn't do option 2 above)
    if !(2*nupdated < npartitions + 1)
        G .= 0
        for ∇i in ∇s
            G .-= ∇i
        end
    end

    # scale the gradient by the number of non-zero partial gradients,
    # important for the first few iterations when some entries of ∇s may be zero
    ∇ .= G
    s = npartitions / (npartitions - sum(iszero, uepochs))
    if !isone(s) && !isinf(s)
        ∇ .*= s
    end

    # uepochs, ∇s
    uepochs, partition_worker_map, G, ∇s    
end

update_gradient!(args...; variancereduced::Bool, kwargs...) = variancereduced ? update_gradient_vr!(args...; kwargs...) : update_gradient_sgd!(args...; kwargs...)

function update_iterate!(V, ∇; state=nothing, stepsize, kwargs...)
    size(V) == size(∇) || throw(DimensionMismatch("V has dimensions $(size(B)), but ∇ has dimensions $(size(∇))"))
    V .-= stepsize .* (∇ .+ V)
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