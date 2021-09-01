export TreeGradient, nsamples

mutable struct TreeGradient{T}
    ∇::T        # overall gradient (sum of all gradients in t)
    n::Int      # total number of samples
    ninit::Int  # number of initialized samples
    t::DataStructures.BalancedTree23{Int,T,Base.Order.ForwardOrdering}
end

function TreeGradient(∇::T, n::Integer) where T
    t = DataStructures.BalancedTree23{Int,T,Base.Order.ForwardOrdering}(Base.Order.ForwardOrdering())
    insert!(t, 1, ∇, false)
    TreeGradient{T}(copy(∇), n, n, t)
end

Base.size(tg::TreeGradient) = size(tg.∇)

function Base.iterate(t::DataStructures.BalancedTree23, state=DataStructures.beginloc(t))
    if state < 3
        return nothing
    end
    (t.data[state].k, t.data[state].d), DataStructures.nextloc0(t, state)
end

Base.iterate(tg::TreeGradient) = iterate(tg.t)
Base.iterate(tg::TreeGradient, state) = iterate(tg.t, state)

function Base.collect(t::DataStructures.BalancedTree23{Tk,Td}) where {Tk,Td}
    rv = Vector{Tuple{Tk,Td}}()
    for v in t
        push!(rv, v)
    end
    rv
end

Base.collect(tg::TreeGradient) = collect(tg.t)

initialized_fraction(tg::TreeGradient) = tg.ninit / tg.n

"""

Insert the gradient with respect to samples `i` through `j`, with value `∇i`, into `t`. If 
`isempty` is `false`, the inserted gradient is counted towards the number of initialized samples.
"""
function Base.insert!(tg::TreeGradient{T}, i::Integer, j::Integer, ∇i::T, isempty::Bool=false) where T
    0 < i <= j <= tg.n || throw(ArgumentError("invalid (i, j): $((i, j))"))
    size(∇i) == size(tg) || throw(DimensionMismatch("∇i has dimensions $(size(∇i)), but existing data has dimensions $(size(tg))"))

    prev, _ = DataStructures.findkey(tg.t, i)
    prev_key = tg.t.data[prev].k
    prev_val = tg.t.data[prev].d
    k, _ = DataStructures.findkey(tg.t, j)
    next = DataStructures.nextloc0(tg.t, k)
    next_key = next == 2 ? tg.n+1 : tg.t.data[next].k

    # remove the previous value from the total, and add the new value to be inserted
    tg.∇ .-= prev_val
    if !isempty
        tg.∇ .+= ∇i
    end

    # update the count of initialized samples
    tg.ninit = max(tg.ninit - (next_key - prev_key), 0)
    tg.ninit = min(tg.ninit+j-i+1, tg.n)

    # delete any intermediate nodes
    k = DataStructures.nextloc0(tg.t, prev)
    key = k == 2 ? tg.n+1 : tg.t.data[k].k
    ∇r1, ∇r2 = ∇i, ∇i # store pointers to data allocated to deleted nodes, so that we can recycle the memory
    while key < next_key
        ∇r1 = ∇r2
        ∇r2 = tg.t.data[k].d
        tg.∇ .-= tg.t.data[k].d
        delete!(tg.t, k)        
        k = DataStructures.nextloc0(tg.t, prev)
        key = k == 2 ? tg.n+1 : tg.t.data[k].k
    end

    # either zero out the node to the left and insert a new node, or, if i matches the start of an
    # existing node, update that node in-place (to avoid mutating the tree unnecessarily)
    if prev_key < i
        prev_val .= 0

        # if a node was deleted, we can re-use the memory allocated for it
        if !(∇r2 === ∇i)
            ∇r2 .= ∇i
        else
            ∇r2 = copy(∇i)
        end
        insert!(tg.t, i, ∇r2, false)
    else
        prev_val .= ∇i
    end    
    if j < next_key - 1

        # if at least 2 nodes were deleted, we can re-use memory also here
        if !(∇r1 === ∇i)
            ∇r1 .= 0
        else
            ∇r1 = zero(∇i)
        end
        insert!(tg.t, j+1, ∇r1, false)
    end
    DataStructures.findkeyless(tg.t, i)
end