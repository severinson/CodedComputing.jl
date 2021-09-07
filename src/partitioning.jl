export partition, translate_partition, align_partitions

"""
    partition(n::Integer, p::Integer, i::Integer)

Divide the integers from `1` to `n` into `p` evenly sized partitions, and return a `UnitRange` 
making up the integers of the `i`-th partition.
"""
function partition(n::Integer, p::Integer, i::Integer)
    0 < n || throw(ArgumentError("n must be positive, but is $n"))
    0 < p <= n || throw(ArgumentError("p must be in [1, $n], but is $p"))    
    0 < i <= p || throw(ArgumentError("i must be in [1, $p], but is $i"))
    (div((i-1)*n, p)+1):div(i*n, p)
end

"""
    translate_partition(n::Integer, p::Integer, q::Integer, i::Integer)

Find `j` such that the `j`-th out of `q` partitions contains the first element contained in the
`i`-th out of `p` partitions.
"""
function translate_partition(n::Integer, p::Integer, q::Integer, i::Integer)
    0 < n || throw(ArgumentError("n must be positive, but is $n"))        
    0 < p <= n || throw(ArgumentError("p must be in [1, $n], but is $p"))        
    0 < q <= n || throw(ArgumentError("q must be in [1, $n], but is $q"))
    0 < i <= p || throw(ArgumentError("i must be in [1, $p], but is $i"))    
    ceil(Int, first(partition(n, p, i)) / n * q)
end

"""
    align_partitions(n::Integer, p::Integer, q::Integer, i::Integer)

Find the largest `j` smaller than or equal to `translate_partition(n, p, q, i)` such that the first
element of the `j`-th out of `q` partitions is equal to the first element of the `i`-th out of `p`
partitions.
"""
function align_partitions(n::Integer, p::Integer, q::Integer, i::Integer)
    0 < n || throw(ArgumentError("n must be positive, but is $n"))    
    0 < p <= n || throw(ArgumentError("p must be in [1, $n], but is $p"))        
    0 < q <= n || throw(ArgumentError("q must be in [1, $n], but is $q"))
    0 < i <= p || throw(ArgumentError("i must be in [1, $p], but is $i"))
    j = translate_partition(n, p, q, i)
    i = translate_partition(n, q, p, j)
    while first(partition(n, q, j)) != first(partition(n, p, i))                
        j -= 1
        i = translate_partition(n, q, p, j)
    end
    j
end