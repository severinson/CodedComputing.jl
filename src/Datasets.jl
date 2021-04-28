"""
    load_genome_data(chr; directory="./1000genomes/parsed")

Read parsed plain-text genome data and return it as a boolean `SparseMatrixCSC`, where the 
`i, j`-th entry indicates if there is a mutation in the `j`-th position of the genom of the `i`-th 
subject (i.e., rows correspond to subjects and columns to genome positions).
"""
function load_genome_data(chr; directory="./1000genomes/parsed")
    filename = joinpath(directory, "chr$(chr).txt")
    df = DataFrame(CSV.File(filename))
    sparse(df[:, 2], df[:, 1], true)
end

"""

Read parsed plain-text genome data for all chromosomes and write it to a single matrix in order.
"""
function genome_to_hdf5(filename="./1000genomes/parsed/1000genomes.h5")
    println("Parsing chromosome 1")
    X = load_genome_data(1)
    h5writecsc(filename, "X", X)
    for chr in 2:22
        GC.gc() # force garbage collection to avoid running out of memory
        println("Parsing chromosome $chr")
        h5appendcsc(filename, "X", load_genome_data(chr))
    end
end

"""

Write a randomly generated sparse matrix.
"""
function write_sprand_matrix(m, n, p, filename, name; nblocks::Integer=ceil(Int, n/10000), overwrite=true)
    firstcol = 1
    lastcol = floor(Int, 1/nblocks*n)
    println("Block 1 / $nblocks")
    h5open(filename, "cw") do fid
        h5writecsc(filename, name, sprand(m, lastcol-firstcol+1, p); overwrite)
        for i in 2:nblocks
            firstcol = floor(Int, (i-1)/nblocks*n+1)
            lastcol = floor(Int, i/nblocks*n)
            println("Block $i / $nblocks")
            h5appendcsc(filename, name, sprand(m, lastcol-firstcol+1, p))
            GC.gc()
        end
    end
    return
end