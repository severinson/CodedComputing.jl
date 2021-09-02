using ArgParse, Random, SparseArrays, HDF5, H5Sparse, SAG

const FROM_WORKER_METADATA_BYTES = 8
const ELEMENT_TYPE = Float32
const CANARY_VALUE = UInt16(2^16 - 1)

function update_argsettings!(s::ArgParseSettings)
    @add_arg_table s begin
        "--iteratedataset"
            help = "Initial iterate dataset name (an initial iterate is selected at random if not provided)"
            arg_type = String
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
    parsed_args[:algorithm] = "pca.jl"
    nworkers::Int = parsed_args[:nworkers]
    nreplicas::Int = parsed_args[:nreplicas]
    mod(nworkers, nreplicas) == 0 || throw(ArgumentError("nworkers is $nworkers, but must be divisible by nreplicas"))
    npartitions = div(nworkers, nreplicas)
    parsed_args[:nwait] = isnothing(parsed_args[:nwait]) ? npartitions : parsed_args[:nwait]

    # record the number of rows and columns of the dataset
    nrows, ncolumns = problem_size(parsed_args[:inputfile], parsed_args[:inputdataset])
    parsed_args[:nrows] = nrows # problem dimension
    parsed_args[:ncolumns] = ncolumns # number of samples

    parsed_args
end

"""

Called inside `asyncmap!` to determine if enough workers have responded. Returns `true` if at 
least `nwait` workers have responded and `false` otherwise.
"""
function fwait(epoch, repochs; nworkers, nwait, nwaitschedule, kickstart, kwargs...)
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

function read_localdata(i::Integer, nworkers::Integer; inputfile::String, inputdataset::String, nreplicas::Integer, kwargs...)
    HDF5.ishdf5(inputfile) || throw(ArgumentError("$inputfile isn't an HDF5 file"))
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    0 < nreplicas || throw(DomainError(nreplicas, "nreplicas must be positive"))
    0 < i <= nworkers || throw(DomainError(i, "i must be in [1, nworkers]"))
    mod(nworkers, nreplicas) == 0 || throw(ArgumentError("nworkers must be divisible by nreplicas"))
    npartitions = div(nworkers, nreplicas)
    partition_index = ceil(Int, i/nreplicas)
    _, nsamples = problem_size(inputfile, inputdataset)
    h5open(inputfile, "r") do fid
        inputdataset in keys(fid) || throw(ArgumentError("$inputdataset is not in $fid"))
        
        # determine the indices of the samples to be stored locally
        Is = partition(nsamples, npartitions, partition_index)

        # load those samples into memory
        if H5Sparse.h5isvalidcsc(fid, inputdataset)
            return sparse(H5SparseMatrixCSC(fid, inputdataset, :, Is))::SparseMatrixCSC
        else
            return fid[inputdataset][:, Is]::Matrix
        end
    end
end

function worker_setup(rank::Integer, nworkers::Integer; ncomponents::Integer, kwargs...)
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))    
    isnothing(ncomponents) || 0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))
    localdata = read_localdata(rank, nworkers; kwargs...)
    dimension = size(localdata, 1)
    to_worker_metadata_bytes = sizeof(UInt16) * 2 * nworkers
    recvbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*ncomponents + to_worker_metadata_bytes)
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*ncomponents + COMMON_BYTES + FROM_WORKER_METADATA_BYTES)
    @info "(rank $rank) setup finished"
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
    to_worker_metadata_bytes = sizeof(UInt16) * 2 * nworkers
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*dimension*ncomponents + to_worker_metadata_bytes)
    recvbuf = Vector{UInt8}(undef, (sizeof(ELEMENT_TYPE)*dimension*ncomponents + COMMON_BYTES + FROM_WORKER_METADATA_BYTES) * nworkers)
    reinterpret(ELEMENT_TYPE, view(sendbuf, (to_worker_metadata_bytes+1):length(sendbuf))) .= view(V, :)

    V, recvbuf, sendbuf
end

metadata_view(buffer) = view(buffer, COMMON_BYTES+1:(COMMON_BYTES + FROM_WORKER_METADATA_BYTES))
data_view(buffer) = reinterpret(ELEMENT_TYPE, @view buffer[(COMMON_BYTES + FROM_WORKER_METADATA_BYTES+1):end])

function worker_task!(recvbuf, sendbuf, localdata; state=nothing, ncomponents::Integer, nworkers::Integer, kwargs...)
    0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))
    to_worker_metadata_bytes = sizeof(UInt16) * 2 * nworkers
    dimension, nlocalsamples = size(localdata)

    # get the number of sub-partitions and the index of the sub-partition to process
    vs = reinterpret(Tuple{UInt16,UInt16}, view(recvbuf, 1:to_worker_metadata_bytes))
    rank <= length(vs) || throw(DimensionMismatch("vs has length $(length(vs)), but rank is $rank"))
    nsubpartitions, subpartition_index = vs[rank]
    0 < nsubpartitions <= nlocalsamples || throw(DimensionMismatch("nsubpartitions is $nsubpartitions, but nlocalsamples is $nlocalsamples"))
    0 < subpartition_index <= nsubpartitions || throw(ArgumentError("subpartition_index is $subpartition_index, but nsubpartitions is $nsubpartitions"))

    # format the recvbuf into a matrix we can operate on
    recvdata = view(recvbuf, (to_worker_metadata_bytes+1):length(recvbuf))
    length(reinterpret(ELEMENT_TYPE, recvdata)) == dimension*ncomponents || throw(DimensionMismatch("recvdata has length $(length(reinterpret(ELEMENT_TYPE, recvdata))), but the data dimension is $dimension and ncomponents is $ncomponents"))
    V = reshape(reinterpret(ELEMENT_TYPE, recvdata), dimension, ncomponents)

    # prepare working memory
    if isnothing(state) # first iteration
        max_samples = nlocalsamples # max number of samples processed per iteration
        W = Matrix{eltype(V)}(undef, max_samples, ncomponents)
    else # subsequent iterations
        W::Matrix{eltype(V)} = state
    end

    # indices of the local samples to process in this iteration
    cols = partition(nlocalsamples, nsubpartitions, subpartition_index)

    # perform the computation
    tcolsmul!(W, localdata, V, cols)
    colsmul!(V, localdata, W, cols)
    V .*= -1
    
    # populate the send buffer
    metadata = reinterpret(UInt16, metadata_view(sendbuf))
    metadata[1] = CANARY_VALUE
    metadata[2] = rank
    metadata[3] = nsubpartitions
    metadata[4] = subpartition_index
    data_view(sendbuf) .= view(V, :)
    W
end

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

update_gradient!(args...; variancereduced::Bool, kwargs...) = variancereduced ? update_gradient_vr!(args...; kwargs...) : update_gradient_sgd!(args...; kwargs...)

function update_iterate!(V, ∇; state=nothing, stepsize, kwargs...)
    size(V) == size(∇) || throw(DimensionMismatch("V has dimensions $(size(V)), but ∇ has dimensions $(size(∇))"))
    V .-= stepsize .* (∇ .+ V)
    orthogonal!(V)
    return
end

function coordinator_task!(V, ∇, recvbufs, sendbuf, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, nworkers::Integer, kwargs...)
    isnothing(state) || length(state) == 2 || throw(ArgumentError("expected state to be nothing or a tuple of length 2, but got $state"))
    if isnothing(state)
        gradient_state = update_gradient!(∇, recvbufs, epoch, repochs; kwargs...)
        iterate_state = update_iterate!(V, ∇; kwargs...)
    else
        gradient_state, iterate_state = state
        gradient_state = update_gradient!(∇, recvbufs, epoch, repochs; state=gradient_state, kwargs...)
        iterate_state = update_iterate!(V, ∇; state=iterate_state, kwargs...)
    end
    to_worker_metadata_bytes = sizeof(UInt16) * 2 * nworkers
    reinterpret(ELEMENT_TYPE, view(sendbuf, (to_worker_metadata_bytes+1):length(sendbuf))) .= view(V, :)
    gradient_state, iterate_state
end

include("common.jl")