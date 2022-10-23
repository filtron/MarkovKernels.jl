using MarkovKernels
using Plots, Random

include("sampling_implementation.jl")
include("sample_trajectory.jl")

state_plt = plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = "t",
    labels = ["x1" "x2"],
    title = ["Latent Gauss-Markov process" ""],
)
display(state_plt)

output_plot = plot(ts, outs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
display(output_plot)
