export ConcurrentCircularBuffer

"""

Thread-safe `CircularBuffer{T}`. Can be used to implement a fixed-capacity FIFO/FILO message queue
with preemption, i.e., pushing a message to a full queue causes a messages already in the queue to
be discarded.
"""
struct ConcurrentCircularBuffer{T}
    cb::CircularBuffer{T}           # underlying circular buffer for storage
    lk::Threads.SpinLock            # for protecting access to the buffer
    c::Threads.Condition            # used to notify threads waiting to pop from the buffer when elements are added
end

ConcurrentCircularBuffer{T}(n::Integer) where T = ConcurrentCircularBuffer{T}(CircularBuffer{T}(n), Threads.SpinLock(), Threads.Condition())

function DataStructures.isfull(ch::ConcurrentCircularBuffer)
    lock(ch.lk) do
        return isfull(ch.cb)
    end
end

function DataStructures.isempty(ch::ConcurrentCircularBuffer)
    lock(ch.lk) do
        return isempty(ch.cb)
    end
end

function DataStructures.empty!(ch::ConcurrentCircularBuffer)
    lock(ch.lk) do
        return empty!(ch.cb)
    end
end

DataStructures.capacity(ch::ConcurrentCircularBuffer) = capacity(ch.cb)
Base.eltype(ch::ConcurrentCircularBuffer) = eltype(ch.cb)

function Base.length(ch::ConcurrentCircularBuffer)
    lock(ch.lk) do
        return length(ch.cb)
    end
end
Base.size(ch::ConcurrentCircularBuffer) = length(ch)

"""

Return `true` if there is at least one element in the buffer.
"""
Base.isready(ch::ConcurrentCircularBuffer) = length(ch) > 0

"""

Add an element to the back and overwrite front if full.
"""
function Base.push!(ch::ConcurrentCircularBuffer, v)
    lock(ch.lk) do
        push!(ch.cb, v)
    end
    lock(ch.c) do
        notify(ch.c)
    end
    ch    
end

"""

Add an element to the front and overwrite back if full.
"""
function DataStructures.pushfirst!(ch::ConcurrentCircularBuffer, v)
    lock(ch.lk) do
        pushfirst!(ch.cb, v)        
    end
    lock(ch.c) do
        notify(ch.c)
    end
    ch
end

"""

Add all elements in `vs` to the back and overwrite front if full.
"""
function Base.append!(ch::ConcurrentCircularBuffer, vs::AbstractVector)
    lock(ch.lk) do
        append!(ch.cb, vs)
    end
    lock(ch.c) do
        notify(ch.c)
    end
    ch
end

"""

Remove the element at the back.
"""
function DataStructures.pop!(ch::ConcurrentCircularBuffer{T})::T where T
    local rv::T
    success = false
    lock(ch.c) do
        while !success
            lock(ch.lk) do
                if length(ch.cb) > 0
                    success = true
                    rv = pop!(ch.cb)
                end
            end
            if !success
                wait(ch.c)
            end
        end
    end
    rv
end

"""

Remove the element at the front.
"""
function DataStructures.popfirst!(ch::ConcurrentCircularBuffer{T})::T where T
    local rv::T
    success = false    
    lock(ch.c) do
        while !success
            lock(ch.lk) do
                if length(ch.cb) > 0
                    success = true
                    rv = popfirst!(ch.cb)
                end
            end
            if !success
                wait(ch.c)
            end
        end
    end
    rv    
end

"""

Same as `popfirst!`.
"""
Base.take!(ch::ConcurrentCircularBuffer) = popfirst!(ch)

function _pop!(vs::AbstractVector, ch::ConcurrentCircularBuffer)
    rv = 0
    for i in 1:length(vs)
        if length(ch.cb) == 0
            break
        end
        vs[i] = pop!(ch.cb)
        rv += 1            
    end
    rv
end

function _popfirst!(vs::AbstractVector, ch::ConcurrentCircularBuffer)
    rv = 0
    for i in 1:length(vs)
        if length(ch.cb) == 0
            break
        end
        vs[i] = popfirst!(ch.cb)
        rv += 1
    end
    rv
end

"""

Remove up to `length(vs)` element at the back, place them in `vs`, and return the number of 
elements removed.
"""
function DataStructures.pop!(vs::AbstractVector, ch::ConcurrentCircularBuffer)
    rv = 0
    success = false
    lock(ch.c) do
        while !success
            lock(ch.lk) do
                if length(ch.cb) > 0
                    success = true
                    rv = _pop!(vs, ch)
                end
            end
            if !success
                wait(ch.c)
            end
        end
    end
    rv
end

"""

Remove up to `length(vs)` element at the front, place them in `vs`, and return the number of 
elements removed.
"""
function DataStructures.popfirst!(vs::AbstractVector, ch::ConcurrentCircularBuffer)
    rv = 0
    success = false
    lock(ch.c) do
        while !success
            lock(ch.lk) do
                if length(ch.cb) > 0
                    success = true
                    rv = _popfirst!(vs, ch)
                end
            end
            if !success
                wait(ch.c)
            end
        end
    end
    rv
end