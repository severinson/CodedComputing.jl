using SparseArrays
export h5writecsc, h5readcsc, isvalidh5csc

function h5writecsc(filename, name::AbstractString, data::SparseMatrixCSC)
    h5open(filename, "cw") do fid
        if name in keys(fid)
            delete!(fid, name)
        end
        g = create_group(fid, name)
        g["colptr"] = data.colptr
        g["m"] = data.m
        g["n"] = data.n
        g["nzval"] = data.nzval
        g["rowval"] = data.rowval
    end
    return
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

Read the `col`-th column of the `SparseMatrixCSC` matrix stored in file `fid` under `name`
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

Read the submatrix consisting of columns `firstcol:lastcol` from `fid`.
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