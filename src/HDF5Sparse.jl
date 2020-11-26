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

function h5readcsc(filename, name::AbstractString)::SparseMatrixCSC
    HDF5.ishdf5(filename) || throw(ArgumentError("$filename isn't a valid HDF5 file"))    
    h5open(filename, "r") do fid
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
end