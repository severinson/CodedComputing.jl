using ArgParse, Random, SparseArrays, HDF5, H5Sparse, SAG

const METADATA_BYTES = 8
const ELEMENT_TYPE = Float32
const CANARY_VALUE = UInt16(2^16 - 1)

include("gradients.jl")

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
        "--vralgo"
            help = "Specifies which variance-reduced optimizer to use (tree or table)"
            arg_type = String
            default = "tree"
            range_tester = (x) -> x in ["tree", "table"]            
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
    0 < parsed_args[:nwait] <= npartitions || throw(ArgumentError("nwait is $(parsed_args[:nwait]), but nworkers/nreplicas is $npartitions"))    

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

function read_localdata(i::Integer, nworkers::Integer; inputfile::String, inputdataset::String, labeldataset::String, nreplicas::Integer, kwargs...)
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
        labeldataset in keys(fid) || throw(ArgumentError("$labeldataset is not in $fid"))
        
        # determine the indices of the samples to be stored locally
        Is = partition(nsamples, npartitions, partition_index)

        # load those samples into memory
        labels = fid[labeldataset][Is]
        if H5Sparse.h5isvalidcsc(fid, inputdataset)
            return sparse(H5SparseMatrixCSC(fid, inputdataset, :, Is))::SparseMatrixCSC, labels
        else
            return fid[inputdataset][:, Is]::Matrix, labels
        end
    end
end

function worker_setup(rank::Integer, nworkers::Integer; kwargs...)
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    features, labels = read_localdata(rank, nworkers; kwargs...)
    dimension = size(features, 1)
    localdata = (features, labels)
    to_worker_metadata_bytes = sizeof(UInt16) * 2 * nworkers
    recvbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*(dimension+1) + to_worker_metadata_bytes)
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*(dimension+1) + COMMON_BYTES + METADATA_BYTES)
    localdata, recvbuf, sendbuf
end

function coordinator_setup(nworkers::Integer; inputfile::String, inputdataset::String, iteratedataset, parsed_args...)    
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    dimension, nsamples = problem_size(inputfile, inputdataset)

    # initial iterate
    V = zeros(ELEMENT_TYPE, dimension+1)
    if !isnothing(iteratedataset) # given as an argument and loaded from disk
        h5open(inputfile) do fid
            iteratedataset in keys(fid) || throw(ArgumentError("iterate dataset $iteratedataset not found"))
            size(fid[iteratedataset]) == (dimension+1,) || throw(DimensionMismatch("Expected iterate to have dimensions $((dimension+1,)), but it has dimensions $(size(fid[iteratedataset]))"))
            V .= fid[iteratedataset][:]
        end
    end

    # communication buffers
    to_worker_metadata_bytes = sizeof(UInt16) * 2 * nworkers
    sendbuf = Vector{UInt8}(undef, sizeof(ELEMENT_TYPE)*(dimension+1) + to_worker_metadata_bytes)
    recvbuf = Vector{UInt8}(undef, (sizeof(ELEMENT_TYPE)*(dimension+1) + COMMON_BYTES + METADATA_BYTES) * nworkers)
    reinterpret(ELEMENT_TYPE, view(sendbuf, (to_worker_metadata_bytes+1):length(sendbuf))) .= view(V, :)

    V, recvbuf, sendbuf, nothing
end

metadata_view(buffer) = view(buffer, COMMON_BYTES+1:(COMMON_BYTES + METADATA_BYTES))
data_view(buffer) = reinterpret(ELEMENT_TYPE, @view buffer[(COMMON_BYTES + METADATA_BYTES+1):end])

function worker_task!(recvbuf, sendbuf, localdata; state=nothing, ncolumns::Integer, nworkers::Integer, kwargs...)
    to_worker_metadata_bytes = sizeof(UInt16) * 2 * nworkers
    features, labels = localdata
    size(features, 2) == length(labels) || throw(DimensionMismatch("features has dimensions $(size(features)), but labels has dimension $(size(labels))"))
    dimension, nlocalsamples = size(features)

    # prepare working memory
    if isnothing(state) # first iteration
        nsubpartitions_prev = 0
        subpartition_index = 1
        w = zeros(ELEMENT_TYPE, nlocalsamples) # temp. storage
    else # subsequent iterations
        nsubpartitions_prev::Int, subpartition_index::Int, w::Vector{ELEMENT_TYPE} = state
    end

    # get the number of sub-partitions from the coordinator
    vs = reinterpret(Tuple{UInt16,UInt16}, view(recvbuf, 1:to_worker_metadata_bytes))
    rank <= length(vs) || throw(DimensionMismatch("vs has length $(length(vs)), but rank is $rank"))
    nsubpartitions, _ = vs[rank]
    
    # increment the sub-partition index
    if !iszero(nsubpartitions_prev)
        subpartition_index = mod(subpartition_index, nsubpartitions_prev) + 1
        subpartition_index = align_partitions(nlocalsamples, nsubpartitions_prev, nsubpartitions, subpartition_index)
    end
    nsubpartitions_prev = nsubpartitions
    0 < nsubpartitions <= nlocalsamples || throw(DimensionMismatch("nsubpartitions is $nsubpartitions, but nlocalsamples is $nlocalsamples"))
    0 < subpartition_index <= nsubpartitions || throw(ArgumentError("subpartition_index is $subpartition_index, but nsubpartitions is $nsubpartitions"))

    # format the recvbuf into a matrix we can operate on
    recvdata = view(recvbuf, (to_worker_metadata_bytes+1):length(recvbuf))
    length(reinterpret(ELEMENT_TYPE, recvdata)) == dimension+1 || throw(DimensionMismatch("recvdata has length $(length(reinterpret(ELEMENT_TYPE, recvdata))), but the data dimension is $dimension"))
    v = reinterpret(ELEMENT_TYPE, recvdata)

    # indices of the local samples to process in this iteration
    cols = partition(nlocalsamples, nsubpartitions, subpartition_index)

    # gradient computation
    tcolsmul!(w, features, view(v, 2:length(v)), cols)
    @views w[cols] .+= v[1] # implicit intercept
    @views w[cols] .*= labels[cols]
    @views w[cols] .= exp.(w[cols])
    @views w[cols] .+= 1
    @views w[cols] .= labels[cols] ./ w[cols]
    @views w[cols] .*= -1
    @views v[1] = sum(w[cols]) # intercept derivative
    colsmul!(view(v, 2:length(v)), features, w, cols)
    v ./= ncolumns # normalize by the total number of samples

    # populate the send buffer
    metadata = reinterpret(UInt16, metadata_view(sendbuf))
    metadata[1] = CANARY_VALUE
    metadata[2] = rank
    metadata[3] = nsubpartitions
    metadata[4] = subpartition_index
    data_view(sendbuf) .= v
    nsubpartitions_prev, subpartition_index, w
end

function update_gradient!(args...; variancereduced::Bool, vralgo::String, kwargs...)
    if variancereduced
        if vralgo == "table"
            return update_gradient_vr!(args...; kwargs...)
        elseif vralgo == "tree"
            return update_gradient_vrt!(args...; kwargs...)
        else
            throw(ArgumentError("unexpected vralgo $vralgo"))
        end
    else
        return update_gradient_sgd!(args...; kwargs...)
    end
end

function update_iterate!(v, ∇; state=nothing, stepsize, lambda, kwargs...)
    size(v) == size(∇) || throw(DimensionMismatch("v has dimensions $(size(v)), but ∇ has dimensions $(size(∇))"))
    v .*= 1 - stepsize * lambda
    v .-= stepsize .* ∇
    return
end

function coordinator_task!(V, ∇, recvbufs, sendbuf, epoch::Integer, repochs::Vector{<:Integer}; nworkers::Integer, state=nothing, kwargs...)
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