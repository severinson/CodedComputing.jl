using ArgParse, Random, SparseArrays, HDF5, H5Sparse, SAG

const METADATA_BYTES = 6
const ELEMENT_TYPE = Float32
const CANARY_VALUE = UInt16(2^16 - 1)

function update_argsettings!(s::ArgParseSettings)
    @add_arg_table s begin
        "--lambda"
            help = "L2 regularizer coefficient"
            arg_type = Float64
            default = 0.0
        "--labeldataset"
            help = "Label dataset name"
            default = "b"
            arg_type = String
        "--iteratedataset"
            help = "Initial iterate dataset name (chosen to be all-zeros if not provided)"
            arg_type = String 
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
        "--nreplicas"
            help = "Number of replicas of each data partition"
            default = 1
            arg_type = Int
            range_tester = (x) -> x >= 1            
        "--nwait"
            help = "Number of replicas to wait for in each iteration (defaults to all replicas)"
            arg_type = Int
            range_tester = (x) -> x >= 1
        "--nwaitschedule"
            help = "Factor by which nwait is reduced per iteration"
            arg_type = Float64
            default = 1.0
            range_tester = (x) -> 0 < x <= 1            
        "--kickstart"
            help = "Wait for all partitions in the first iteration"
            action = :store_true            
    end
end

function update_parsed_args!(s::ArgParseSettings, parsed_args)
    parsed_args[:algorithm] = "logreg.jl"
    nworkers::Int = parsed_args[:nworkers]
    nreplicas::Int = parsed_args[:nreplicas]
    mod(nworkers, nreplicas) == 0 || throw(ArgumentError("nworkers is $nworkers, but must be divisible by nreplicas"))
    npartitions = div(nworkers, nreplicas)
    parsed_args[:nwait] = isnothing(parsed_args[:nwait]) ? npartitions : parsed_args[:nwait]

    # record the number of rows and columns of the dataset
    nrows, ncolumns = problem_size(parsed_args[:inputfile], parsed_args[:inputdataset])
    parsed_args[:nrows] = nrows
    parsed_args[:ncolumns] = ncolumns

    parsed_args
end

"""

Called inside `asyncmap!` to determine if enough workers have responded. Returns `true` if at 
least `nwait` workers have responded and `false` otherwise.
"""
function fwait(epoch, repochs; nworkers, nwait, kickstart, nwaitschedule, kwargs...)
    length(repochs) == nworkers || throw(DomainError(nworkers, "repochs must have length nworkers"))
    0 < nwait <= nworkers || throw(ArgumentError("nwait is $nwait, but must be in [1, npartitions]"))
    nwait = min(max(1, ceil(Int, nwait*nwaitschedule^epoch)), nworkers)
    nrec = 0
    for repoch in repochs
        nrec += repoch == epoch
    end
    if epoch == 1 && kickstart
        return nrec == nwait
    else
        return nrec >= nwait
    end
end

"""

Return the size of the data matrix.
"""
function problem_size(filename::String, dataset::String)
    HDF5.ishdf5(filename) || throw(ArgumentError("$filename isn't an HDF5 file"))
    h5open(filename, "r") do fid
        dataset in keys(fid) || throw(ArgumentError("$dataset is not in $fid"))
        if H5Sparse.h5isvalidcsc(fid, dataset)
            return size(H5SparseMatrixCSC(fid, dataset))
        else
            return size(fid[dataset])
        end
    end
end

function partition_samples(X::H5SparseMatrixCSC, nsubpartitions::Integer)
    nsamples = size(X, 2)
    dividers = round.(Int, range(1, nsamples+1, length=nsubpartitions+1))
    [sparse(X[:, dividers[i]:(dividers[i+1]-1)]) for i in 1:nsubpartitions]
end

function partition_samples(X::Matrix, nsubpartitions::Integer)
    nsamples = size(X, 2)
    dividers = round.(Int, range(1, nsamples+1, length=nsubpartitions+1))
    [view(X, :, dividers[i]:(dividers[i+1]-1)) for i in 1:nsubpartitions]    
end

function partition_samples(v::Vector, nsubpartitions::Integer)
    nsamples = length(v)
    dividers = round.(Int, range(1, nsamples+1, length=nsubpartitions+1))
    [view(v, dividers[i]:(dividers[i+1]-1)) for i in 1:nsubpartitions]    
end

function read_localdata(i::Integer, nworkers::Integer; inputfile::String, inputdataset::String, labeldataset::String, nreplicas::Integer, nsubpartitions::Integer, kwargs...)
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
        size(fid[labeldataset]) == (nsamples,) || throw(DimensionMismatch("Labels has dimensions $(size(fid[labeldataset])), but there are $nsamples samples"))
        # read nreplicas/nworkers samples
        il = floor(Int, (partition_index - 1)/npartitions*nsamples + 1)
        iu = floor(Int, partition_index/npartitions*nsamples)        
        labels = partition_samples(fid[labeldataset][il:iu], nsubpartitions)
        if H5Sparse.h5isvalidcsc(fid, inputdataset) # sparse data
            X_sparse = H5SparseMatrixCSC(fid, inputdataset, :, il:iu)
            return partition_samples(X_sparse, nsubpartitions), labels, dimension, nsamples
        else # dense data
            X_dense = fid[inputdataset][:, il:iu]
            return partition_samples(X_dense, nsubpartitions), labels, dimension, nsamples
        end
    end
end

function worker_setup(rank::Integer, nworkers::Integer; kwargs...)
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    features, labels, dimension, nsamples = read_localdata(rank, nworkers; kwargs...)
    localdata = (features, labels)
    recvbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*(dimension+1))
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*(dimension+1) + COMMON_BYTES + METADATA_BYTES)
    localdata, recvbuf, sendbuf
end

function coordinator_setup(nworkers::Integer; inputfile::String, inputdataset::String, iteratedataset, parsed_args...)    
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    dimension, nsamples = problem_size(inputfile, inputdataset)

    # initial iterate
    if isnothing(iteratedataset)
        V = zeros(dimension+1)
    else # given as an argument and loaded from disk
        h5open(inputfile) do fid
            iteratedataset in keys(fid) || throw(ArgumentError("iterate dataset $iteratedataset not found"))
            size(fid[iteratedataset]) == (dimension+1,) || throw(DimensionMismatch("Expected iterate to have dimensions $((dimension+1,)), but it has dimensions $(size(fid[iteratedataset]))"))
            V = fid[iteratedataset][:]
        end
    end

    # communication buffers
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*(dimension+1))
    recvbuf = Vector{UInt8}(undef, (sizeof(ELEMENT_TYPE)*(dimension+1) + COMMON_BYTES + METADATA_BYTES) * nworkers)
    reinterpret(ELEMENT_TYPE, view(sendbuf, :)) .= view(V, :)

    V, recvbuf, sendbuf
end

metadata_view(buffer) = view(buffer, COMMON_BYTES+1:(COMMON_BYTES + METADATA_BYTES))
data_view(buffer) = reinterpret(ELEMENT_TYPE, @view buffer[(COMMON_BYTES + METADATA_BYTES+1):end])

function worker_task!(recvbuf, sendbuf, localdata; state=nothing, nsubpartitions::Integer, ncolumns::Integer, kwargs...)
    sizeof(recvbuf) + COMMON_BYTES + METADATA_BYTES == sizeof(sendbuf) || throw(DimensionMismatch("recvbuf has size $(sizeof(recvbuf)), but sendbuf has size $(sizeof(sendbuf))"))
    feature_partitions, label_partitions = localdata
    length(feature_partitions) == nsubpartitions || throw(DimensionMismatch("There are $(length(feature_partitions)) feature partitions, but nsubpartitions is $nsubpartitions"))
    length(label_partitions) == nsubpartitions || throw(DimensionMismatch("There are $(length(label_partitions)) label partitions, but nsubpartitions is $nsubpartitions"))
    
    # select a random sub-partition
    subpartition_index = rand(1:nsubpartitions)
    Xw = feature_partitions[subpartition_index]
    bw = label_partitions[subpartition_index]
    dimension, nlocalsamples = size(Xw)
    1 <= nsubpartitions <= dimension || throw(DimensionMismatch("nsubpartitions is $nsubpartitions, but the dimension is $dimension"))
    length(bw) == size(Xw, 2) || throw(DimensionMismatch("bw has dimension $(length(bw)), but Xw has dimensions $(size(Xw))"))

    # format the recvbuf into a matrix we can operate on
    v = reinterpret(ELEMENT_TYPE, recvbuf)
    length(v) == dimension+1 || throw(DimensionMismatch("v has dimension $(length(v)), but the data dimension is $(dimension+1)"))    

    # prepare working memory
    if isnothing(state) # first iteration
        max_samples = maximum((x)->length(x), label_partitions) # max number of samples processed per iteration
        w = Vector{eltype(v)}(undef, max_samples) # temp. storage
    else # subsequent iterations
        w::Vector{eltype(v)} = state
    end

    # compute gradient
    wv = view(w, 1:nlocalsamples)
    mul!(wv', view(v, 2:length(v))', Xw)
    wv .+= v[1] # implicit intercept
    wv .*= bw
    wv .= exp.(wv)
    wv .+= 1
    wv .= bw ./ wv
    wv .*= -1
    v[1] = sum(wv) # intercept derivative
    mul!(view(v, 2:length(v), :), Xw, wv) # derivative w. respect to each feature
    v ./= ncolumns # normalize by the total number of samples
    
    # populate the send buffer
    metadata = reinterpret(UInt16, metadata_view(sendbuf))
    metadata[1] = CANARY_VALUE
    metadata[2] = rank
    metadata[3] = subpartition_index
    data_view(sendbuf) .= v
    w
end

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
        ∇ .+= Vw
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

update_gradient!(args...; variancereduced::Bool, kwargs...) = variancereduced ? update_gradient_vr!(args...; kwargs...) : update_gradient_sgd!(args...; kwargs...)

function update_iterate!(v, ∇; state=nothing, stepsize, lambda, kwargs...)
    size(v) == size(∇) || throw(DimensionMismatch("v has dimensions $(size(v)), but ∇ has dimensions $(size(∇))"))
    v .*= 1 - stepsize * lambda
    v .-= stepsize .* ∇
    return
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