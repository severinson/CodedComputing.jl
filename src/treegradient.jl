export TreeGradient, TreeGradientNode

mutable struct TreeGradientNode{T}
   ∇::T
   isinitialized::Bool
end

mutable struct TreeGradient{T}
    ∇::T        # overall gradient (sum of all gradients in t)
    n::Int      # total number of samples
    ninit::Int  # number of initialized samples
    t::DataStructures.BalancedTree23{Int,TreeGradientNode{T},Base.Order.ForwardOrdering}
end

function TreeGradient(∇::T, n::Integer; isempty::Bool=false) where T
    t = DataStructures.BalancedTree23{Int,TreeGradientNode{T},Base.Order.ForwardOrdering}(Base.Order.ForwardOrdering())
    insert!(t, 1, TreeGradientNode(∇, !isempty), false)
    ninit = isempty ? 0 : n
    TreeGradient(copy(∇), n, ninit, t)
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

Return the number of samples that make up the sub-gradient at `tg.t.data[i]`.
"""
function subgradient_nsamples(tg::TreeGradient, i::Integer)
    first_sample = tg.t.data[i].k
    next = DataStructures.nextloc0(tg.t, i)
    final_sample = next == 2 ? tg.n : tg.t.data[next].k-1
    final_sample - first_sample + 1
end

"""

Insert the gradient with respect to samples `i` through `j`, with value `∇i`, into `t`. If 
`isempty` is `false`, the inserted gradient is counted towards the number of initialized samples.
"""
function Base.insert!(tg::TreeGradient{T}, i::Integer, j::Integer, ∇i::T; isempty::Bool=false) where T
    0 < i <= j <= tg.n || throw(ArgumentError("invalid (i, j): $((i, j))"))
    size(∇i) == size(tg) || throw(DimensionMismatch("∇i has dimensions $(size(∇i)), but existing data has dimensions $(size(tg))"))
    prev, _ = DataStructures.findkey(tg.t, i)
    prev_key = tg.t.data[prev].k
    prev_val = tg.t.data[prev].d
    k, _ = DataStructures.findkey(tg.t, j)
    next = DataStructures.nextloc0(tg.t, k)
    next_key = next == 2 ? tg.n+1 : tg.t.data[next].k

    # take out the previous sub-gradient
    if prev_val.isinitialized
        tg.∇ .-= prev_val.∇
        tg.ninit -= subgradient_nsamples(tg, prev)
    end

    # add the new sub-gradient
    if !isempty
        tg.∇ .+= ∇i
        tg.ninit += j - i + 1
    end    


    # delete any intermediate nodes
    k = DataStructures.nextloc0(tg.t, prev)
    key = k == 2 ? tg.n+1 : tg.t.data[k].k
    ∇r1, ∇r2 = ∇i, ∇i # store pointers to data allocated to deleted nodes, so that we can recycle the memory
    while key < next_key
        node = tg.t.data[k]
        ∇r1 = ∇r2
        ∇r2 = node.d.∇
        if node.d.isinitialized
            tg.∇ .-= node.d.∇
            tg.ninit -= subgradient_nsamples(tg, k)
        end
        delete!(tg.t, k)

        k = DataStructures.nextloc0(tg.t, prev)
        key = k == 2 ? tg.n+1 : tg.t.data[k].k
    end

    # either zero out the node to the left and insert a new node, or, if i matches the start of an
    # existing node, update that node in-place (to avoid mutating the tree unnecessarily)
    if prev_key < i
        prev_val.∇ .= 0            
        prev_val.isinitialized = false
        
        # if a node was deleted, we can re-use the memory allocated for it
        if !(∇r2 === ∇i)
            ∇r2 .= ∇i
        else
            ∇r2 = copy(∇i)
        end
        insert!(tg.t, i, TreeGradientNode(∇r2, !isempty), false)
    else
        prev_val.∇ .= ∇i
        prev_val.isinitialized = !isempty
    end    
    if j < next_key - 1

        # if at least 2 nodes were deleted, we can re-use memory also here
        if !(∇r1 === ∇i)
            ∇r1 .= 0
        else
            ∇r1 = zero(∇i)
        end
        insert!(tg.t, j+1, TreeGradientNode(∇r1, false), false)
    end
    DataStructures.findkeyless(tg.t, i)
end