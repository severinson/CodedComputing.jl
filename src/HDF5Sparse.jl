export h5writecsc, h5readcsc

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

function h5readcsc(filename, name::AbstractString)::SparseMatrixCSC
    h5open(filename, "r") do fid
        g = fid[name]
        colptr = g["colptr"][:]
        m = g["m"][]
        n = g["n"][]
        nzval = g["nzval"][:]
        rowval = g["rowval"][:]
        return SparseMatrixCSC(m, n, colptr, rowval, nzval)
    end
end