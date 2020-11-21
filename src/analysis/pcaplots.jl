using CSV, DataFrames, PyPlot, Statistics

read_df(filename="data/pca/iterationtime.csv") = DataFrame(CSV.File(filename, normalizenames=true))

"""

Plot the CCDF of the iteration time for all values of `nwait` for the given number of workers.
"""
function plot_iterationtime_cdf(df; nworkers::Integer=9)
    df = df[df.iteration .>= 10, :]
    df = df[df.nworkers .== nworkers, :]
    plt.figure()
    for nwait in sort!(unique(df.nwait))
        df_nwait = df[df.nwait .== nwait, :]
        x = sort(df_nwait.t_compute)
        y = 1 .- range(0, 1, length=length(x))
        plt.loglog(x, y, label="($nworkers, $nwait")
    end
    plt.ylim(1e-2, 1)
    plt.xlim(1e-4, 1e-2)
    plt.xlabel("Iteration time [s]")
    plt.ylabel("CCDF")
    plt.grid()
    plt.legend()
    plt.show()
end

"""

Plot the `q`-th quantile of the iteration time as a function of `nwait`, i.e., 
the number of workers waited for in each iteration.
"""
function plot_iterationtime(df; nworkers::Integer=9, q::Real=0.5)
    df = df[df.iteration .>= 10, :]
    df = df[df.nworkers .== nworkers, :]
    x = 1:nworkers
    y = [quantile(df[df.nwait .== nwait, :].t_compute, q) for nwait in x]
    plt.figure()
    plt.plot(x, y)
    plt.grid()
    plt.xlim(1, nworkers)
    plt.xlabel("nwait")
    plt.ylabel("Iteration time, $(q)-th quantile [s]")
end