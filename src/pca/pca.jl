function problem_size(filename::String, dataset::String)
    h5open(filename, "r") do file
        return size(file[dataset])
    end
end

function read_localdata(filename::String, dataset::String, i::Integer, npartitions::Integer)
    h5open(filename, "r") do file
        n, m = size(file[dataset])
        il = round(Int, (i - 1)/npartitions*n + 1)
        iu = round(Int, i/npartitions*n)
        return file[dataset][il:iu, :]
    end
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
    ∇ .= 0
    nresults = 0
    for (V, repoch) in zip(Vs, repochs)        
        if repoch == epoch
            ∇ .+= V
            nresults += 1
        end
    end
    ∇ .*= length(Vs) / nresults
    state
end

function update_iterate!(V, ∇, state=nothing)
    V .= ∇
    orthogonal!(V)
    state
end

include("common.jl")