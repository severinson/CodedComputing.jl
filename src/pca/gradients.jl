# gradient estimation methods

function update_gradient_sgd!(∇, recvbufs, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, nreplicas, ncolumns::Integer, kwargs...)
    length(recvbufs) == length(repochs) || throw(DimensionMismatch("recvbufs has dimension $(length(recvbufs)), but repochs has dimension $(length(repochs))"))
    0 < nreplicas || throw(DomainError(nreplicas, "nreplicas must be positive"))
    epoch <= 1 || !isnothing(state) || error("expected state to be initiated for epoch > 1")
    nworkers = length(recvbufs)
    mod(nworkers, nreplicas) == 0 || throw(ArgumentError("nworkers must be divisible by nreplicas"))

    # record the epoch at which each partition was last updated
    if isnothing(state)
        uepochs = Dict{Int,Int}()
    else
        uepochs::Dict{Int,Int} = state
    end

    # add at most 1 replica of each partition to the overall gradient
    # the partitions are arranged sequentially, so if there are 2 partitions and 3 replicas, then
    # Vs is of length 6, and its elements correspond to partitions [1, 1, 1, 2, 2, 2]
    ∇ .= 0
    processed_nsamples = 0
    for worker_index in 1:nworkers

        # skip workers that we've never received anything from
        if repochs[worker_index] == 0
            continue
        end

        metadata = reinterpret(UInt16, metadata_view(recvbufs[worker_index]))
        if length(metadata) != 4
            @error "received incorrectly formatted metadata from the $(worker_index)-th worker in epoch $epoch: $metadata"
            continue
        end
        canary, worker_rank, nsubpartitions, subpartition_index = metadata
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
        if repochs[worker_index] < epoch || (haskey(uepochs, partition_index) && uepochs[partition_index] == epoch)
            continue
        end

        # compute the number of samples that make up this partition, and add it to the total
        worker_nsamples = length(partition(ncolumns, div(nworkers, nreplicas), replica_index))
        subpartition_nsamples = length(partition(worker_nsamples, nsubpartitions, subpartition_index))
        processed_nsamples += subpartition_nsamples

        # add the sub-gradient computed by this worker
        uepochs[partition_index] = epoch
        Vw = reshape(data_view(recvbufs[worker_index]), size(∇)...)
        ∇ .+= Vw
    end

    # scale the (stochastic) gradient to make it unbiased estimate of the true gradient
    processed_fraction = processed_nsamples / ncolumns
    if !(processed_fraction ≈ 1)
        ∇ ./= processed_fraction
    end
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
        worker_from_partition = zeros(Int, npartitions)
        sg = StochasticGradient(eltype(∇), npartitions, size(∇))
    else
        uepochs, worker_from_partition, sg = state
        worker_from_partition .= 0
    end    

    # iterate over the received partial gradients to record which of them are newer than what we currently have
    for worker_index in 1:nworkers

        # skip workers that we've never received anything from
        if repochs[worker_index] == 0
            continue
        end

        metadata = reinterpret(UInt16, metadata_view(recvbufs[worker_index]))
        if length(metadata) != 4
            @error "received incorrectly formatted metadata from the $(worker_index)-th worker in epoch $epoch: $metadata"
            continue
        end
        canary, worker_rank, worker_nsubpartitions, subpartition_index = metadata
        if  canary != CANARY_VALUE
            @error "recieved incorrect canary value from the $(worker_index)-th worker in epoch $epoch: $canary"
            continue
        end
        worker_nsubpartitions == nsubpartitions || throw(ArgumentError("update_gradient_vr doesn't support dynamically changing the number of partitions"))
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
        worker_from_partition[partition_index] = worker_index
        uepochs[partition_index] = repochs[worker_index]        
    end

    # update the gradient with the received sub-gradients
    partition_indices = [i for i in 1:npartitions if !iszero(worker_from_partition[i])]
    worker_indices = [worker_from_partition[i] for i in partition_indices]
    ∇s = [reshape(data_view(recvbufs[worker_index]), size(∇)...) for worker_index in worker_indices]
    update!(sg, zip(partition_indices, ∇s))

    # scale the gradient by the number of initialized sub-gradients
    f = initialized_fraction(sg)
    if isone(f)
        ∇ .= sg
    else
        ∇ .= sg ./ f
    end

    uepochs, worker_from_partition, sg
end

function update_gradient_vrt!(∇, recvbufs, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, nostale::Bool, nreplicas::Integer, ncolumns::Integer, kwargs...)
    length(recvbufs) == length(repochs) || throw(DimensionMismatch("recvbufs has dimension $(length(recvbufs)), but repochs has dimension $(length(repochs))"))
    0 < nreplicas || throw(DomainError(nreplicas, "nreplicas must be positive"))
    epoch <= 1 || !isnothing(state) || error("expected state to be initiated for epoch > 1")
    nworkers = length(recvbufs)
    mod(nworkers, nreplicas) == 0 || throw(ArgumentError("nworkers must be divisible by nreplicas"))
    
    # record the epoch at which each partition was last updated
    # store the previously computed partial gradients
    if isnothing(state)
        uepochs = Dict{Int,Int}()
        ∇i = zero(∇)
        tg = TreeGradient(zero(∇), ncolumns)
    else
        uepochs::Dict{Int,Int}, ∇i::typeof(∇), tg::TreeGradient{typeof(∇)} = state
    end

    # iterate over the received partial gradients to record which of them are newer than what we currently have
    for worker_index in 1:nworkers

        # skip workers that we've never received anything from
        if repochs[worker_index] == 0
            continue
        end

        metadata = reinterpret(UInt16, metadata_view(recvbufs[worker_index]))
        if length(metadata) != 4
            @error "received incorrectly formatted metadata from the $(worker_index)-th worker in epoch $epoch: $metadata"
            continue
        end
        canary, worker_rank, nsubpartitions, subpartition_index = metadata
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

        # skip partitions with no new information
        if haskey(uepochs, partition_index) && repochs[worker_index] <= uepochs[partition_index]
            continue
        end
        uepochs[partition_index] = repochs[worker_index]

        # compute which samples make up this partition
        worker_samples = partition(ncolumns, div(nworkers, nreplicas), replica_index)
        worker_nsamples = length(worker_samples)
        subpartition_samples = first(worker_samples) .+ partition(worker_nsamples, nsubpartitions, subpartition_index) .- 1

        # insert the new gradient
        ∇i .= reshape(data_view(recvbufs[worker_index]), size(∇)...)
        insert!(tg, first(subpartition_samples), last(subpartition_samples), ∇i)
    end

    # scale the gradient by the number of initialized sub-gradients
    fraction_processed = tg.ninit / tg.n
    if isapprox(fraction_processed, 1)
        ∇ .= tg.∇
    else
        ∇ .= tg.∇ ./ fraction_processed
    end

    uepochs, ∇i, tg
end