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

function worker_task!(V, Xw; state=nothing, pfraction=1, kwargs...)
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
    W, p
end

function update_gradient!(∇, Vs, epoch::Integer, repochs::Vector{<:Integer}; state=nothing, pfraction=1, kwargs...)
    length(Vs) == length(repochs) || throw(DimensionMismatch("Vs has dimension $(length(Vs)), but repochs has dimension $(length(repochs))"))
    ∇ .= 0
    nresults = 0
    for (V, repoch) in zip(Vs, repochs)        
        if repoch == epoch
            ∇ .+= V
            nresults += 1
        end
    end
    ∇ .*= length(Vs) / nresults / pfraction
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