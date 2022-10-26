# Implementing a Kalman filter

```@meta
CurrentModule = MarkovKernels
```

### Setting up the environment and loading some data


```@example 2
using MarkovKernels
using Random, LinearAlgebra, Plots

rng = MersenneTwister(1991)

include("../../demo/sampling_implementation.jl")
include("../../demo/sample_trajectory.jl")

output_plot = plot(ts, outs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
```

### Implementing a Kalman filter

```@example 2
function kalman_filter(
    ys::AbstractVecOrMat,
    init::AbstractNormal,
    fw_kernel::AbstractNormalKernel,
    m_kernel::AbstractNormalKernel,
)
    n = size(ys, 1)

    # initialise recursion
    filter_distribution = init
    filter_distributions = Normal[]      # filtering distributions

    # create measurement model
    y = ys[1, :]
    likelihood = LogLike(m_kernel, y)

    # measurement update
    filter_distribution, loglike_increment = bayes_rule(filter_distribution, likelihood)
    push!(filter_distributions, filter_distribution)
    loglike = loglike_increment

    for m in 2:n

        # predict
        filter_distribution = marginalise(filter_distribution, fw_kernel)

        # create measurement model
        y = ys[m, :]
        likelihood = LogLike(m_kernel, y)

        # measurement update
        filter_distribution, loglike_increment = bayes_rule(filter_distribution, likelihood)

        push!(filter_distributions, filter_distribution)
        loglike = loglike + loglike_increment
    end

    return filter_distributions, loglike
end
```

### Computing the state estimates

```@example 2
filter_distributions, loglike = kalman_filter(ys, init, fw_kernel, m_kernel)

state_plt = plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = ["" "t"],
    label = ["x1" "x2"],
    title = ["Filter estimates of the state" ""],
)
plot!(ts, filter_distributions, layout = (2, 1), label = ["x1filter" "x2filter"])

state_plt
```


### Computing the output estimates

```@example 2
output_filter_estimate = map(z -> marginalise(z, output_kernel), filter_distributions)

plt = plot(ts, outs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
plot!(ts, output_filter_estimate, label = "filter estimate")
```