using ArgParse

function update_argsettings!(s::ArgParseSettings)
    @add_arg_table s begin
        "--nminibatches"
            help = "Number of batches that the data stored by each worker is partitioned into. In each iteration, the worker selects one of the partitions at random to compute the gradient."
            arg_type = Int
            default = 1
        "--stepsize"
            help = "Gradient descent step size"
            arg_type = Float64
            default = 1.0
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

function read_localdata(filename::String, dataset::String, i::Integer, npartitions::Integer; kwargs...)
    h5open(filename, "r") do fid
        dataset in keys(fid) || throw(ArgumentError("$dataset is not in $fid"))
        flag, _ = isvalidh5csc(fid, dataset)
        if flag
            X = h5readcsc(fid, dataset)
            m = size(X, 1)
            il = round(Int, (i - 1)/npartitions*m + 1)
            iu = round(Int, i/npartitions*m)
            return X[il:iu, :]            
        else            
            n, m = size(fid[dataset])
            il = round(Int, (i - 1)/npartitions*n + 1)
            iu = round(Int, i/npartitions*n)
            return fid[dataset][il:iu, :]
        end
    end
end

function worker_task!(V, Xw; state=nothing, nminibatches=1, kwargs...)
    0 < nminibatches <= size(Xw, 1) || throw(DomainError(nminibatches, "nminibatches must be in [1, size(Xw, 1)]"))
    if isnothing(state)
        W = Matrix{eltype(V)}(undef, size(Xw, 1), size(V, 2))
    else
        W = state
    end    
    n = size(Xw, 1)
    i = rand(1:nminibatches)
    il = round(Int, (i - 1)/nminibatches*n + 1)
    iu = round(Int, i/nminibatches*n)
    Xwv = view(Xw, il:iu, :)
    Wv = view(W, il:iu, :)
    mul!(Wv, Xwv, V)
    mul!(V, Xwv', Wv)
    W
end

function update_gradient!(∇, Vs, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, nminibatches=1, kwargs...)
    length(Vs) == length(repochs) || throw(DimensionMismatch("Vs has dimension $(length(Vs)), but repochs has dimension $(length(repochs))"))
    ∇ .= 0
    nresults = 0
    for (V, repoch) in zip(Vs, repochs)        
        if repoch == epoch
            ∇ .+= V
            nresults += 1
        end
    end
    ∇ .*= length(Vs) / nresults * nminibatches
    state
end

function update_iterate!(V, ∇; state=nothing, stepsize=1, kwargs...)
    size(V) == size(∇) || throw(DimensionMismatch("V has dimensions $(size(B)), but ∇ has dimensions $(size(∇))"))
    for I in CartesianIndices(V)
        V[I] -= stepsize * (∇[I] + V[I])
    end
    orthogonal!(V)
    state
end

include("common.jl")