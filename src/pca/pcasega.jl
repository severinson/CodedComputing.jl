function update_parsed_args!(s, parsed_args)
    parsed_args["algorithm"] = "pcasega.jl"
end

function problem_size(filename::String, dataset::String)
    HDF5.ishdf5(filename) || throw(ArgumentError("$filename isn't an HDF5 file"))
    h5open(filename, "r") do fid
        flag, msg = isvalidh5csc(fid, dataset)
        if !flag
            throw(ArgumentError(msg))
        end
        g = fid[dataset]
        return g["m"][], g["n"][]
    end
end

function read_localdata(filename::String, dataset::String, i::Integer, npartitions::Integer)
    HDF5.ishdf5(filename) || throw(ArgumentError("$filename isn't an HDF5 file"))
    X = h5readcsc(filename, dataset)
    m = size(X, 1)
    il = round(Int, (i - 1)/npartitions*m + 1)
    iu = round(Int, i/npartitions*m)
    Xw = X[il:iu, :]
    Xw
end

function worker_task!(V, Xw, state=nothing)
    if isnothing(state)
        W = Matrix{eltype(V)}(undef, size(Xw, 1), size(V, 2))
    else
        W = state
    end
    mul!(W, Xw, V)
    mul!(V, Xw', W)
    W
end

function update_gradient!(∇, Vs, epoch::Integer, repochs::Vector{<:Integer}, state=nothing)
    length(Vs) == length(repochs) || throw(DimensionMismatch("Vs has dimension $(length(Vs)), but repochs has dimension $(length(repochs))"))
    nworkers = length(Vs)
    if isnothing(state)
        ∇s = [similar(∇) for _ in 1:nworkers]
        for ∇i in ∇s
            ∇i .= 0
        end
        uepochs = similar(repochs)
        uepochs .= -1
    else
        ∇s, uepochs = state
    end    
    length(uepochs) == length(repochs) || throw(DimensionMismatch("uepochs has dimension $(length(uepochs)), but repochs has dimension $(length(repochs))"))
    ∇ .= 0
    for i in 1:nworkers
        if repochs[i] > uepochs[i]
            ∇s[i] .= Vs[i]
        end
        ∇ .+= ∇s[i]
    end
    ∇s, uepochs
end

function update_iterate!(V, ∇, state=nothing)
    V .= ∇
    orthogonal!(V)
    state
end

include("common.jl")