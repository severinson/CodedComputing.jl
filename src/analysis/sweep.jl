"""

Plot latency samples vs. the number of flops
"""
function plot_sweep(df)
    for density in sort!(unique(df.density))
        dfi = df[df.density .== density, :]
        plt.plot(dfi.nflops, dfi.latency, ".", label="Density: $density")

        # ts = range(0, maximum(dfi.nflops), length=100)
        # p, coeffs = CodedComputing.fit_polynomial(dfi.nflops, dfi.latency, 1)
        # plt.plot(ts, p.(ts), "k--")

        # p, coeffs = CodedComputing.fit_polynomial(dfi.nflops, dfi.latency, 2)
        # plt.plot(ts, p.(ts), "r--")        

        dfj = by(dfi, :nflops, :latency => mean => :mean)
        plt.plot(dfj.nflops, dfj.mean, "o")
    end
    plt.legend()
    plt.grid()
end