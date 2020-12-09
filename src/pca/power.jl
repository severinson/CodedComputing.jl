using ArgParse, Random
using GradientSketching

const METADATA_BYTES = 4
const ELEMENT_TYPE = Float64

function update_argsettings!(s::ArgParseSettings)
    @add_arg_table s begin
        "--npartitions"
            help = "Number of partitions to split the input data into"
            arg_type = Int
            required = true
            range_tester = (x) -> x >= 1
        "--codeweight"
            help = "Codeword weight, i.e., the number of uncoded partitions stored by each worker"
            arg_type = Int
            default = 3
            range_tester = (x) -> x >= 1
    end
end

function update_parsed_args!(s::ArgParseSettings, parsed_args)
    parsed_args[:algorithm] = "power.jl"
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

function read_localdata(i::Integer; inputfile::String, inputdataset::String, npartitions::Integer, kwargs...)
    HDF5.ishdf5(inputfile) || throw(ArgumentError("$inputfile isn't an HDF5 file"))
    0 < i <= npartitions || throw(DomainError(i, "i must be in [1, npartitions]"))
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

codeword_indices(i; npartitions::Integer, codeweight::Integer) = [mod(i+j-2, npartitions)+1 for j in 1:codeweight]
codeword_coefficients(seed; codeweight::Integer) = randn(MersenneTwister(seed), codeweight)

function worker_setup(rank::Integer, nworkers::Integer; codeweight::Integer, npartitions::Integer, ncomponents, kwargs...)
    0 < nworkers || throw(DomainError(nworkers, "nworkers must be positive"))
    isnothing(ncomponents) || 0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))
    1 <= npartitions <= nworkers || throw(DomainError(npartitions, "npartitions must be in [1, nworkers]"))
    1 <= codeweight <= npartitions || throw(DimensionMismatch("codeweight is $codeweight, but npartitions is $npartitions"))

    # load multiple data partitions to store locally
    localdata = [read_localdata(i; npartitions, kwargs...) for i in codeword_indices(rank; npartitions, codeweight)]

    dims = length(size(localdata[1]))
    dims == 2 || error("Expected localdata to be 2-dimensional, but got data of dimension $dims")
    dimension = size(localdata[1], 2)
    for partition in localdata
        size(partition, 2) == dimension || throw(DimensionMismatch("inconsistent data dimension for local data partitions"))
    end

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

function worker_task!(recvbuf, sendbuf, localdata; state=nothing, npartitions::Integer, codeweight::Integer, ncomponents, kwargs...)
    isnothing(ncomponents) || 0 < ncomponents || throw(DomainError(ncomponents, "ncomponents must be positive"))        
    sizeof(recvbuf) + METADATA_BYTES == sizeof(sendbuf) || throw(DimensionMismatch("recvbuf has size $(sizeof(recvbuf)), but sendbuf has size $(sizeof(sendbuf))"))
    1 <= npartitions || throw(DomainError(npartitions, "npartitions must be positive"))
    1 <= codeweight <= npartitions || throw(DimensionMismatch("codeweight is $codeweight, but npartitions is $npartitions"))        
    length(localdata) == codeweight || throw(DimensionMismatch("localdata is $(length(localdata)), but codeweight is $codeweight"))
    dimension = size(localdata[1], 2)

    # default to computing all components
    if isnothing(ncomponents)
        k = dimension
    else
        k = ncomponents
    end

    # format the recvbuf into a matrix we can operate on
    length(reinterpret(ELEMENT_TYPE, recvbuf)) == dimension*k || throw(DimensionMismatch("recvbuf has length $(length(reinterpret(ELEMENT_TYPE, recvbuf))), but the data dimension is $dimension and ncomponents is $k"))
    V = reshape(reinterpret(ELEMENT_TYPE, recvbuf), dimension, k)

    # initialize state
    if isnothing(state)
        max_rows = maximum((x)->size(x, 1), localdata)
        W = Matrix{eltype(V)}(undef, max_rows, size(V, 2)) # intermediate results (first multiplication)
        M = similar(V) # intermediate results (second multiplication)
        C = similar(V) # coded symbol
    else
        W, M, C = state
    end

    # generate a random seed, and use that seed to generate codeword coefficients
    # the seed is sent to the coordinator, which use it to reproduce the same coefficients
    seed = UInt16(rand(0:2^16-1))
    coefficients = codeword_coefficients(seed; codeweight)

    # do the computation
    C .= 0
    for (Xw, coefficient) in zip(localdata, coefficients)
        Wv = view(W, 1:size(Xw, 1), :)
        mul!(Wv, Xw, V)
        mul!(M, Xw', Wv)
        M .*= coefficient
        C .+= M
    end
    V .= C
    
    # populate the send buffer
    metadata = reinterpret(UInt16, view(sendbuf, 1:METADATA_BYTES))
    metadata[1] = seed
    metadata[2] = 0 # unused

    @views sendbuf[METADATA_BYTES+1:end] .= recvbuf[:]
    W, M, C
end

data_view(recvbuf) = reinterpret(ELEMENT_TYPE, @view recvbuf[METADATA_BYTES+1:end])
metadata_view(recvbuf) = view(recvbuf, 1:METADATA_BYTES)

function update_gradient!(∇, recvbufs, sendbuf, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, codeweight::Integer, npartitions::Integer, kwargs...)
    epoch <= 1 || !isnothing(state) || error("expected state to be initiated for epoch > 1")
    length(recvbufs) == length(repochs) || throw(DimensionMismatch("recvbufs has dimension $(length(recvbufs)), but repochs has dimension $(length(repochs))"))    
    nworkers = length(recvbufs)
    1 <= npartitions <= nworkers || throw(DomainError(npartitions, "npartitions must be in [1, nworkers]"))
    1 <= codeweight <= npartitions || throw(DimensionMismatch("codeweight is $codeweight, but npartitions is $npartitions"))

    if isnothing(state)
        G = zeros(Float64, nworkers, npartitions)
        ∇s = [zeros(eltype(∇), size(∇)...) for _ in 1:npartitions]
    else
        G, ∇s = state
    end

    # populate the rows of the generator matrix corresponding to workers that returned in this epoch    
    worker_indices = findall((repoch)->repoch==epoch, repochs)
    for worker_index in worker_indices

        # get the seed used by the worker to generate codeword coefficients
        metadata = reinterpret(UInt16, metadata_view(recvbufs[worker_index]))
        if length(metadata) != 2
            @error "received incorrectly formatted metadata from the $(worker_index)-th worker in epoch $epoch: $metadata"
            continue
        end
        seed, _ = metadata

        # use the seed to re-create the same coefficients that were generated by the worker
        G[worker_index, :] .= 0
        indices = codeword_indices(worker_index; npartitions, codeweight)::Vector{Int}
        coefficients = codeword_coefficients(seed; codeweight)::Vector{Float64}
        length(indices) == length(coefficients) || throw(DimensionMismatch("codeword indices has diemsnion $(length(indices)), but there are $(length(coefficients)) coefficients"))
        G[worker_index, indices] .= coefficients
    end

    # rows of G and received results for the workers that responded in this epoch
    S = view(G, worker_indices, :)'
    Vs = [reshape(data_view(recvbufs[i]), size(∇)...) for i in worker_indices]

    # recover (an approximation of) the elements ∇s 
    project!(∇s, Vs, S)

    # approximate the overall result by the sum of the recovered partial results
    ∇ .= 0
    for ∇i in ∇s
        ∇ .+= ∇i
    end

    G, ∇s
end

function update_iterate!(V, ∇, sendbuf, epoch, repochs; state=nothing, kwargs...)
    size(V) == size(∇) || throw(DimensionMismatch("V has dimensions $(size(B)), but ∇ has dimensions $(size(∇))"))
    V .= ∇
    orthogonal!(V)
    reinterpret(ELEMENT_TYPE, view(sendbuf, :)) .= view(V, :)
    state
end

include("common.jl")