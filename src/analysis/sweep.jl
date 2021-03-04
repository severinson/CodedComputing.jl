"""

Plot latency samples vs. the number of flops
"""
function plot_sweep(df, color="b", label="")
    markers = ["o", "s"]
    for (density, marker) in zip(sort!(unique(df.density)), markers)
        dfi = df[df.density .== density, :]
        plt.plot(dfi.nrows, dfi.latency, color*".")

        ts = range(0, maximum(dfi.nrows), length=100)

        # p, coeffs = CodedComputing.fit_polynomial(dfi.nflops, dfi.latency, 1)
        # plt.plot(ts, p.(ts), "k--")

        p, coeffs = fit_polynomial(dfi.nrows, dfi.latency, 2)
        plt.plot(ts, p.(ts), color*"--")      
        println(coeffs)  

        dfj = by(dfi, :nrows, :latency => mean => :mean)
        plt.plot(dfj.nrows, dfj.mean, color*marker, label=label) # *" (density: $density)"
    end
    plt.legend()
    plt.grid()
    plt.xlabel("n")
    plt.ylabel("Latency")
end