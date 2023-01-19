# Kalman filtering and smoothing

This tutorial describes how to perform filtering and smoothing in a the probabilistic state-space model given by

```math
\begin{aligned}
x_0 &\sim \mathcal{N}(\mu_0 ,\Sigma_0), \\
x_n \mid x_{n-1} &\sim \mathcal{N}(\Phi  x_{n-1}, Q),\\
z_n &= C x_n,
\end{aligned}
```

subject to the measurements given by

```math
y_n \mid x_n \sim \mathcal{N}(Cx_n,R).
```

## Setting up the environment and generating some data

```@example 2
using MarkovKernels
using Random, LinearAlgebra, Plots, IterTools

rng = MersenneTwister(1991)

function sample(rng, init, K, nstep)
    it = Iterators.take(iterated(z -> rand(rng, K, z), rand(rng, init)), nstep + 1)
    return mapreduce(permutedims, vcat, collect(it))
end

# time grid
m = 200
T = 5
ts = collect(LinRange(0, T, m))
dt = T / (m - 1)

# define transtion kernel
λ = 2.0
Φ = exp(-λ * dt) .* [1.0 0.0; -2*λ*dt 1.0]
Q = I - exp(-2 * λ * dt) .* [1.0 -2*λ*dt; -2*λ*dt 1+(2*λ*dt)^2]
fw_kernel = NormalKernel(Φ, Q)

# initial distribution
init = Normal(zeros(2), diagm(ones(2)))

# sample state
xs = sample(rng, init, fw_kernel, m - 1)

# output kernel and measurement kernel
C = 1.0 / sqrt(2) * [1.0 -1.0]
output_kernel = DiracKernel(C)
R = fill(0.1, 1, 1)
m_kernel = compose(NormalKernel(1.0I(1), R), output_kernel)

# sample output and its measurements
zs = mapreduce(z -> rand(rng, output_kernel, xs[z, :]), vcat, 1:m)
ys = mapreduce(z -> rand(rng, m_kernel, xs[z, :]), vcat, 1:m)

output_plot = plot(ts, zs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
```

## Implementing a Kalman filter

```@example 2
function kalman_filter(
    ys::AbstractVecOrMat,
    init::AbstractNormal,
    fw_kernel::AbstractNormalKernel,
    m_kernel::AbstractNormalKernel,
)

    # initialise recursion
    filter_distribution = init
    filter_distributions = typeof(init)[]

    # initial measurement update
    likelihood = LogLike(m_kernel, ys[1, :])
    filter_distribution, loglike_increment = bayes_rule(filter_distribution, likelihood)
    push!(filter_distributions, filter_distribution)
    loglike = loglike_increment

    for m in 2:size(ys, 1)

        # predict
        filter_distribution = marginalize(filter_distribution, fw_kernel)

        # measurement update
        likelihood = LogLike(m_kernel, ys[m, :])
        filter_distribution, loglike_increment = bayes_rule(filter_distribution, likelihood)
        push!(filter_distributions, filter_distribution)
        loglike = loglike + loglike_increment
    end

    return filter_distributions, loglike
end
```

## Computing the filtered state estimates

```@example 2
filter_distributions, loglike = kalman_filter(ys, init, fw_kernel, m_kernel)

state_filter_plt = plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = ["" "t"],
    label = ["x1" "x2"],
    title = ["Filter estimates of the state" ""],
)
plot!(ts, filter_distributions, layout = (2, 1), label = ["x1filter" "x2filter"])
```


## Computing the filtered output estimates

```@example 2
output_filter_estimate = map(z -> marginalize(z, output_kernel), filter_distributions)

output_filter_plt = plot(ts, zs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
plot!(ts, output_filter_estimate, label = "filter estimate")
```

## Implementing a Rauch-Tung-Striebel recursion

```@example 2
function rts(filter_distributions, fw_kernel)
    smoother_distribution = filter_distributions[end]
    smoother_distributions = similar(filter_distributions)
    smoother_distributions[end] = smoother_distribution

    for m in length(smoother_distributions)-1:-1:1
        pred, bw_kernel = invert(filter_distributions[m], fw_kernel)
        smoother_distribution = marginalize(smoother_distribution, bw_kernel)
        smoother_distributions[m] = smoother_distribution
    end

    return smoother_distributions
end
```

## Computing the smoothed state estimate

```@example 2
smoother_distributions = rts(filter_distributions, fw_kernel)

state_smoother_plt = plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = ["" "t"],
    label = ["x1" "x2"],
    title = ["Smoother estimates of the state" ""],
)
plot!(ts, smoother_distributions, layout = (2, 1), label = ["x1smoother" "x2smoother"])
```

## Computing the smoothed output estimate
```@example 2
output_smoother_estimate = map(z -> marginalize(z, output_kernel), smoother_distributions)
output_smoother_plt = plot(ts, zs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
plot!(ts, output_smoother_estimate, label = "smoother estimate")
```