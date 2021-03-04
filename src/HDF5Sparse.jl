using SparseArrays
export h5writecsc, h5appendcsc, h5readcsc, isvalidh5csc, h5permutecsc

"""
    h5appendcsc(fid::HDF5.File, name::AbstractString, data::SparseMatrixCSC)

Append the matrix `data` to the right of the `SparseMatrixCSC` already stored in `fid[name]`.
"""
function h5appendcsc(fid::HDF5.File, name::AbstractString, data::SparseMatrixCSC)
    name in keys(fid) || throw(ArgumentError("$name doesn't exist in $fid"))
    g = fid[name]
    m, n = g["m"][], g["n"][]
    m == size(data, 1) || throw(DimensionMismatch("Existing array has dimensions $((m, n)), but the new array has dimensions $(size(data))"))

    colptr = data.colptr
    i = size(g["colptr"], 1)
    j = i + length(colptr) - 1
    HDF5.set_dims!(g["colptr"], (j,))
    offset = size(g["rowval"], 1)
    g["colptr"][i:j]  = colptr .+ offset

    nzval = nonzeros(data)
    i = size(g["nzval"], 1) + 1
    j = i + length(nzval) - 1
    HDF5.set_dims!(g["nzval"], (j,))
    g["nzval"][i:j]  = nzval

    rowval = rowvals(data)
    i = size(g["rowval"], 1) + 1
    j = i + length(rowval) - 1
    HDF5.set_dims!(g["rowval"], (j,))
    g["rowval"][i:j]  = rowval

    delete_object(g, "n")
    g["n"] = n + size(data, 2)
    return
end

"""
    h5appendcsc(filename::AbstractString, args...; kwargs...)
"""
function h5appendcsc(filename::AbstractString, args...; kwargs...)
    h5open(filename, "cw") do fid
        return h5appendcsc(fid, args...; kwargs...)
    end
end

"""
    h5writecsc(fid::HDF5.File, name::AbstractString, data::SparseMatrixCSC; overwrite=false, batchsize=1000)

Write the matrix `data` to `fid[data]`, overwriting any existing dataset with the same name if `overwrite=true`.
"""
function h5writecsc(fid::HDF5.File, name::AbstractString, data::SparseMatrixCSC; overwrite=false, batchsize=100000, blosc=5)
    if name in keys(fid)
        if overwrite
            delete_object(fid, name)
        else
            throw(ArgumentError("Dataset with name $name already exists"))
        end
    end
    g = create_group(fid, name)
    g["m"] = data.m
    g["n"] = data.n
    
    colptr = data.colptr
    g_colptr = create_dataset(
        g, "colptr", 
        eltype(colptr), 
        ((length(colptr),), (-1,)),
        chunk=(batchsize,),
        blosc=blosc,        
    )
    g_colptr[1:length(colptr)] = colptr

    nzval = nonzeros(data)
    g_nzval = create_dataset(
        g, "nzval", 
        eltype(nzval), 
        ((length(nzval),), (-1,)),
        chunk=(batchsize,),
        blosc=blosc,
    )
    g_nzval[1:length(nzval)] = nzval
    
    rowval = rowvals(data)
    g_rowval = create_dataset(
        g, "rowval", 
        eltype(rowval), 
        ((length(rowval),), (-1,)),
        chunk=(batchsize,),
        blosc=blosc,        
    )
    g_rowval[1:length(rowval)] = rowval
    return
end

function h5writecsc(filename::AbstractString, args...; kwargs...)
    h5open(filename, "cw") do fid
        return h5writecsc(fid, args...; kwargs...)
    end
end

function isvalidh5csc(fid::HDF5.File, name::AbstractString)
    s = "Invalid CSC HDF5 file: "
    name in keys(fid) || return false, s*"$name isn't a member of $fid"
    g = fid[name]
    g isa HDF5.Group || return false, s*"expected $name to be a group, but it is a $(typeof(g))"
    for key in ["m", "n", "colptr", "nzval", "rowval"]
        key in keys(g) || return false, s*"expected $key to be a member of $g"
    end
    true, ""
end

"""
    h5readcsc(fid::HDF5.File, name::AbstractString)

Read the `SparseMatrixCSC` matrix stored in `fid[name]`.
"""
function h5readcsc(fid::HDF5.File, name::AbstractString)::SparseMatrixCSC
    flag, msg = isvalidh5csc(fid, name)
    if !flag
        throw(ArgumentError(msg))
    end
    g = fid[name]
    colptr = g["colptr"][:]
    m = g["m"][]
    n = g["n"][]
    nzval = g["nzval"][:]
    rowval = g["rowval"][:]
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

"""

Read the `col`-th column of the `SparseMatrixCSC` matrix stored in `fid[name]`.
"""
function h5readcsc(fid::HDF5.File, name::AbstractString, col::Integer)::SparseVector
    g = fid[name]
    m = g["m"][]
    n = g["n"][]
    0 < col <= n || throw(BoundsError((fid, name), col))
    i = g["colptr"][col]
    j = g["colptr"][col+1] - 1
    rowval = g["rowval"][i:j]     
    nzval = g["nzval"][i:j]
    SparseVector(m, rowval, nzval)
end

"""

Read the submatrix consisting of columns `firstcol:lastcol` from `fid[name]`.
"""
function h5readcsc(fid::HDF5.File, name::AbstractString, firstcol::Integer, lastcol::Integer)::SparseMatrixCSC
    g = fid[name]
    m = g["m"][]
    n = g["n"][]
    0 < firstcol <= n || throw(BoundsError((fid, name), firstcol))    
    0 < lastcol <= n || throw(BoundsError((fid, name), lastcol))    
    firstcol <= lastcol || throw(ArgumentError("expected firstcol <= lastcol"))
    colptr = g["colptr"][firstcol:lastcol+1]
    i = colptr[1]
    j = colptr[end] - 1
    rowval = g["rowval"][i:j]     
    nzval = g["nzval"][i:j]
    colptr .-= i-1
    SparseMatrixCSC(m, lastcol-firstcol+1, colptr, rowval, nzval)
end

function h5readcsc(filename::AbstractString, args...; kwargs...)
    HDF5.ishdf5(filename) || throw(ArgumentError("$filename isn't a valid HDF5 file"))    
    h5open(filename, "r") do fid
        return h5readcsc(fid, args...; kwargs...)
    end
end

"""
    h5permutecsc(srcfid::HDF5.File, srcname::AbstractString, dstfid::HDF5.File, dstname::AbstractString, p::AbstractVector{<:Integer}; overwrite=false)

Split the `SparseMatrixCSC` stored in `srcfid[srcname]` column-wise into `length(p)` partitions and 
write those partitions to `dstfid[dstname]` in the order specified by the permutation vector `p`.
"""
function h5permutecsc(srcfid::HDF5.File, srcname::AbstractString, dstfid::HDF5.File, dstname::AbstractString, p::AbstractVector{<:Integer}; overwrite=false)
    srcg = srcfid[srcname]
    dstg = srcfid[srcname]
    m, n = srcg["m"][], srcg["n"][]
    1 < length(p) <= n || throw(DimensionMismatch("p has dimension $(length(p)), but the source matrix has dimensions $((m, n))"))
    nblocks = length(p)
    i = p[1]
    firstcol = round(Int, (i-1)/nblocks*n+1)
    lastcol = round(Int, i/nblocks*n)  
    X = h5readcsc(srcfid, srcname, firstcol, lastcol)
    h5writecsc(dstfid, dstname, X; overwrite)
    for i in view(p, 2:length(p))
        firstcol = round(Int, (i-1)/nblocks*n+1)
        lastcol = round(Int, i/nblocks*n)
        X = h5readcsc(srcfid, srcname, firstcol, lastcol)
        h5appendcsc(dstfid, dstname, X)
        GC.gc() # force GC to make sure we don't run out of memory        
    end
end

function h5permutecsc(srcfile::AbstractString, srcname::AbstractString, dstfile::AbstractString, dstname::AbstractString, args...; kwargs...)
    srcfid = h5open(srcfile, "cw")
    dstfid = dstfile == srcfile ? srcfid : h5open(dstfile, "cw")
    try
        h5permutecsc(srcfid, srcname, dstfid, dstname, args...; kwargs...)
    finally
        close(srcfid)
        if dstfid != srcfid
            close(dstfid)
        end
    end
end