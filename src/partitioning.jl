export partition

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