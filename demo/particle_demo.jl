# Bootstrap filtering and smoothing

## Setting up the environment and generating some data

using MarkovKernels
using Plots, Random
using IterTools
import StatsBase: wsample

rng = MersenneTwister(1991)

function sample(rng, init, K, nstep)
    it = Iterators.take(iterated(z -> rand(rng, K, z), rand(rng, init)), nstep + 1)
    return mapreduce(permutedims, vcat, collect(it))
end

# time grid
m = 50
T = 5
ts = collect(LinRange(0, T, m))
dt = T / (m - 1)

# define transtion kernel
λ = 1.0
Φ = exp(-λ * dt) .* [1.0 0.0; -2*λ*dt 1.0]
Q = I - exp(-2 * λ * dt) .* [1.0 -2*λ*dt; -2*λ*dt 1+(2*λ*dt)^2]
fw_kernel = NormalKernel(Φ, Q)

# initial distribution
init = Normal(zeros(2), 1.0I(2))

# sample state
xs = sample(rng, init, fw_kernel, m - 1)

# output kernel
σ = 5.0
C = σ / sqrt(2) * [1.0 -1.0]

output_kernel = DiracKernel(C)

variance(x) = fill(exp.(x)[1], 1, 1)
m_kernel = compose(NormalKernel(zeros(1, 1), variance), output_kernel)

# sample output
zs = mapreduce(z -> rand(rng, output_kernel, xs[z, :]), vcat, 1:m)
ys = mapreduce(z -> rand(rng, m_kernel, xs[z, :]), vcat, 1:m)

measurement_plt = scatter(ts, ys, label = "measurements", color = "black")
display(measurement_plt)

## Implementing a bootstrap filter

function bootstrap_filter(
    rng::AbstractRNG,
    ys::AbstractVecOrMat,
    init::AbstractDistribution,
    fw_kernel::AbstractMarkovKernel,
    m_kernel::AbstractMarkovKernel,
    K::Integer,
)

    # initialize
    X = [rand(rng, init) for k in 1:K]
    P = ParticleSystem(zeros(K), X)
    loglike = 0.0
    L = Likelihood(m_kernel, ys[1, :])
    loglike_incr = bayes_rule!(P, L)
    loglike = loglike + loglike_incr
    resample!(rng, P)

    Ps = [P]
    sizehint!(Ps, size(ys, 1))

    for m in 2:size(ys, 1)
        L = Likelihood(m_kernel, ys[m, :])
        P = predict(rng, P, fw_kernel)
        loglike_incr = bayes_rule!(P, L)
        loglike = loglike + loglike_incr
        resample!(rng, P)
        push!(Ps, P)
    end

    return Ps, loglike
end

function resample!(rng::AbstractRNG, P::ParticleSystem{T,U,<:AbstractVector}) where {T,U}
    idx = wsample(rng, eachindex(logweights(P)), weights(P), nparticles(P))
    logweights(P)[:] .= zero(logweights(P))
    particles(P)[:] .= particles(P)[idx]
end

function predict(
    rng::AbstractRNG,
    P::ParticleSystem{T,U,<:AbstractVector},
    K::AbstractMarkovKernel,
) where {T,U}
    X = [rand(rng, K, particles(P)[i]) for i in eachindex(particles(P))]

    return ParticleSystem(copy(logweights(P)), X)
end

## Computing the filtered state estimates

K = 500
Pfilt, loglike_filt = bootstrap_filter(rng, ys, init, fw_kernel, m_kernel, K)

Xfilt = mapreduce(permutedims, vcat, particles.(Pfilt))

state_filt_plt = plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = "t",
    labels = ["x1" "x2"],
    title = ["Latent Gauss-Markov process" ""],
)
for k in 1:K
    scatter!(
        ts,
        mapreduce(permutedims, vcat, Xfilt[:, k]),
        markersize = 1,
        color = "red",
        alpha = 0.025,
        label = "",
    )
end
display(state_filt_plt)

## Computing the filtered output estimates

bf_output_filt = [marginalize(Pfilt[i], output_kernel) for i in eachindex(Pfilt)]
Zfilt = getindex.(mapreduce(permutedims, vcat, particles.(bf_output_filt)), 1)

output_filt_plt = plot(ts, zs, label = "output", xlabel = "t", title = "log-variance")
scatter!(ts, Zfilt, markersize = 1, color = "red", alpha = 0.01, label = "")
display(output_filt_plt)

## Implementing a bootstrap smoother

function bootstrap_smoother(
    rng::AbstractRNG,
    ys::AbstractVecOrMat,
    init::AbstractDistribution,
    fw_kernel::AbstractMarkovKernel,
    m_kernel::AbstractMarkovKernel,
    K::Integer,
)

    # initialize
    X = permutedims([rand(rng, init) for k in 1:K])
    P = ParticleSystem(zeros(K), X)
    loglike = 0.0
    L = Likelihood(m_kernel, ys[1, :])
    loglike_incr = bayes_rule!(P, L)
    loglike = loglike + loglike_incr
    resample!(rng, P)

    for m in 2:size(ys, 1)
        L = Likelihood(m_kernel, ys[m, :])

        P = predict(rng, P, fw_kernel)
        loglike_incr = bayes_rule!(P, L)

        loglike = loglike + loglike_incr
        resample!(rng, P)
    end

    return P, loglike
end

function resample!(rng::AbstractRNG, P::ParticleSystem{T,U,<:AbstractMatrix}) where {T,U}
    idx = wsample(rng, eachindex(logweights(P)), weights(P), nparticles(P))
    logweights(P)[:] .= zero(logweights(P))
    particles(P)[:, :] .= particles(P)[:, idx]
end

function predict(
    rng::AbstractRNG,
    P::ParticleSystem{T,U,<:AbstractMatrix},
    K::AbstractMarkovKernel,
) where {T,U}
    X = [rand(rng, K, particles(P)[end, i]) for i in 1:nparticles(P)]

    return ParticleSystem(logweights(P), vcat(particles(P), permutedims(X)))
end

## Computing the smoothed state estimates

Psmooth, loglike_smooth = bootstrap_smoother(rng, ys, init, fw_kernel, m_kernel, K)

Xsmooth = particles(Psmooth)

state_smooth_plt = plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = "t",
    labels = ["x1" "x2"],
    title = ["Latent Gauss-Markov process" ""],
)
for k in 1:K
    plot!(
        ts,
        mapreduce(permutedims, vcat, Xsmooth[:, k]),
        color = "green",
        alpha = 0.025,
        label = "",
    )
end
display(state_smooth_plt)

## Computing the smoothed output estimate

bf_output_smooth = marginalize(Psmooth, output_kernel)
Zsmooth = getindex.(particles(bf_output_smooth), 1)

output_smooth_plt = plot(ts, zs, label = "output", xlabel = "t", title = "log-variance")
plot!(ts, Zsmooth, color = "green", alpha = 0.025, label = "")
display(output_smooth_plt)
