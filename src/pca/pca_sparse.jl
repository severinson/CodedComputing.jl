function problem_size(filename::String, dataset::String)
    print("getting problem size")
    h5open(filename, "r") do fid        
        return fid[dataset]["m"][], fid[dataset]["n"][]
    end
end

function read_localdata(filename::String, dataset::String, i::Integer, npartitions::Integer)
    print("reading local data\n")
    m, n = problem_size(filename, dataset)
    print("foo $((m, n))\n")
    il = round(Int, (i - 1)/npartitions*m + 1)
    iu = round(Int, i/npartitions*m)    
    X = h5readcsc(filename, dataset)
    Xw = X[il:iu, :]
    print("bar $(size(Xw))\n")
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

function update_gradient!(∇, Vs, bs::AbstractVector{<:Bool}, state=nothing)
    length(Vs) == length(bs) || throw(DimensionMismatch("Vs has dimension $(length(Vs)), but bs has dimension $(length(bs))"))
    ∇ .= 0
    nresults = 0
    for (b, V) in zip(bs, Vs)
        if b
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