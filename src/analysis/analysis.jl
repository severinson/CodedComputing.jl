using CSV
using DataFrames
using PyPlot

# filename = "data/Fri Nov 13 10-55-27 UTC 2020.csv"
# df = DataFrame(CSV.File(filename, normalizenames=true))
function plot_ccdf(df)
    for nworkers in unique(df.nworkers)
        df_nworkers = df[df.nworkers .== nworkers, :]
        if "nwait" in names(df)
            for nwait in [1, 5, 9]
                df_nwait = df_nworkers[df_nworkers.nwait .== nwait, :]
                rtts = sort(df_nwait.rtt)
                rtts .*= 1e6
                ccdf = 1 .- range(0, length(rtts), length=length(rtts)) ./ length(rtts)
                plt.semilogy(rtts, ccdf, label="Async ($nworkers, $nwait)")
            end
        else
            rtts = sort(df_nworkers.rtt)
            rtts .*= 1e6
            ccdf = 1 .- range(0, length(rtts), length=length(rtts)) ./ length(rtts)
            plt.semilogy(rtts, ccdf, label="Bcast ($nworkers)")
        end
    end

end

function plot_ccdf(dfs::AbstractVector)
    plt.figure()
    for df in dfs
        plot_ccdf(df)
    end
    plt.xlabel("Time [μs]")
    plt.ylabel("RTT CCDF")
    plt.ylim(1/10000, 1)
    plt.xlim(0, 300)
    plt.grid()
    plt.legend()    
    plt.show()
end