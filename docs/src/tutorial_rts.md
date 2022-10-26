# Implementing a Rauch-Tung-Striebeel smoother

```@meta
CurrentModule = MarkovKernels
```

### Setting up the environment and loading some data

```@example 3
using MarkovKernels
using Random, LinearAlgebra, Plots

rng = MersenneTwister(1991)

include("../../demo/sampling_implementation.jl")
include("../../demo/sample_trajectory.jl")
include("../../demo/kalman_filter_implementation.jl")

output_plot = plot(ts, outs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
```


### Implementing a Rauch-Tung-Striebel smoother

```@example 3
function rts(
    ys::AbstractVecOrMat,
    init::AbstractNormal,
    fw_kernel::AbstractNormalKernel,
    m_kernel::AbstractNormalKernel,
)
    # run a Kalman filter
    filter_distributions, loglike = kalman_filter(ys, init, fw_kernel, m_kernel)

    # compute the backward kenrles for the Rauch-Tung-Striebel recursion
    bw_kernels = NormalKernel[]
    for m in 1:length(filter_distributions)-1
        pred, bw_kernel = invert(filter_distributions[m], fw_kernel)
        push!(bw_kernels, bw_kernel)
    end

    # compute the smoother estimates
    smoother_distributions = Normal[]
    smoother_distribution = filter_distributions[end]
    pushfirst!(smoother_distributions, smoother_distribution)
    for m in length(filter_distributions)-1:-1:1
        smoother_distribution = marginalise(smoother_distribution, bw_kernels[m])
        pushfirst!(smoother_distributions, smoother_distribution)
    end

    return smoother_distributions, filter_distributions, loglike
end
```


### Computing state estimates

```@example 3
# Rauch-Tung-Striebel smoother
smoother_distributions, filter_distributions, loglike =
    rts(ys, init, fw_kernel, m_kernel)

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
plot!(ts, smoother_distributions, layout = (2, 1), label = ["x1smoother" "x2smoother"])
```

### Computing the output estimates

```@example 3
output_filter_estimate = map(z -> marginalise(z, output_kernel), filter_distributions)
output_smoother_estimate = map(z -> marginalise(z, output_kernel), smoother_distributions)

plt = plot(ts, outs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
plot!(ts, output_filter_estimate, label = "filter estimate")
plot!(ts, output_smoother_estimate, label = "smooher estimate")
```