using MarkovKernels
using Plots, Random

rng = MersenneTwister(1991)

include("sampling_implementation.jl")
include("kalman_filter_implementation.jl")
include("sample_trajectory.jl")

# kalman filter 
filter_distributions, loglike = kalman_filter(ys, init, fw_kernel, m_kernel)

# plotting the filter state estimates
state_plt = plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = ["" "t"],
    label = ["x1" "x2"],
    title = ["Filter estimates of the state" ""],
)
plot!(ts, filter_distributions, layout = (2, 1), label = ["x1filter" "x2filter"])

display(state_plt)

output_filter_estimate = map(z -> marginalise(z, output_kernel), filter_distributions)

plt = plot(ts, outs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
plot!(ts, output_filter_estimate, label = "filter estimate")