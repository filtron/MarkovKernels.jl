using MarkovKernels
using LinearAlgebra, Plots, Random, IterTools

## sample a partially observed Gauss-Markov process

rng = MersenneTwister(1991)

function sample(rng, init, K, nstep)
    it = Iterators.take(iterated(z -> rand(rng, K, z), rand(rng, init)), nstep + 1)
    return mapreduce(permutedims, vcat, collect(it))
end

# time grid
m = 200
T = 5.0
ts = collect(LinRange(0.0, T, m))
dt = T / (m - 1)

# define transtion kernel
λ = 2.0
Φ = exp(-λ * dt) .* [1.0 0.0; -2*λ*dt 1.0]
Q = 1.0I - exp(-2 * λ * dt) .* [1.0 -2*λ*dt; -2*λ*dt 1.0+(2*λ*dt)^2]
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

## implement a the Kalman filter

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
    likelihood = Likelihood(m_kernel, ys[1, :])
    filter_distribution, loglike_increment = bayes_rule(filter_distribution, likelihood)
    push!(filter_distributions, filter_distribution)
    loglike = loglike_increment

    for m in 2:size(ys, 1)

        # predict
        filter_distribution = marginalize(filter_distribution, fw_kernel)

        # measurement update
        likelihood = Likelihood(m_kernel, ys[m, :])
        filter_distribution, loglike_increment = bayes_rule(filter_distribution, likelihood)
        push!(filter_distributions, filter_distribution)
        loglike = loglike + loglike_increment
    end

    return filter_distributions, loglike
end

## run Kalman filter and plot the results

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

display(state_filter_plt)

## computing the output estimates

output_filter_estimate = map(z -> marginalize(z, output_kernel), filter_distributions)

output_filter_plt = plot(ts, zs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
plot!(ts, output_filter_estimate, label = "filter estimate")
display(output_filter_plt)

## implement Rauch-Tung-Striebel recursion

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

## Computing the snmoothed state estimate

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

display(state_smoother_plt)

## Computing the smoothed output estimate

output_smoother_estimate = map(z -> marginalize(z, output_kernel), smoother_distributions)
output_smoother_plt = plot(ts, zs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
plot!(ts, output_smoother_estimate, label = "smoother estimate")
display(output_smoother_plt)
