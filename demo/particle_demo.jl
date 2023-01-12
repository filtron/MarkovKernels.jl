using MarkovKernels
using Plots, Random
using IterTools

#rng = MersenneTwister(1991)

rng = Random.GLOBAL_RNG

include("sampling_implementation.jl")

# time grid
m = 100
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
outs = mapreduce(z -> rand(rng, output_kernel, xs[z, :]), vcat, 1:m)
ys = mapreduce(z -> rand(rng, m_kernel, xs[z, :]), vcat, 1:m)

mplot = scatter(ts, ys, label = "measurement", color = "black")
display(mplot)

include("bootstrap_filter.jl")

K = 100

Pfilt, loglike_filt = bootstrap_filter(rng, ys, init, fw_kernel, m_kernel, K)

Xfilt = mapreduce(permutedims, vcat, particles.(Pfilt))

bf_output_filt = [marginalise(Pfilt[i], output_kernel) for i in eachindex(Pfilt)]
Yfilt = getindex.(mapreduce(permutedims, vcat, particles.(bf_output_filt)), 1)

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
        alpha = 0.05,
        label = "",
    )
end
display(state_filt_plt)

output_filt_plt = plot(ts, outs, label = "output", xlabel = "t", title = "log-variance")
scatter!(ts, Yfilt, markersize = 1, color = "red", alpha = 0.01, label = "")
display(output_filt_plt)

include("bootstrap_smoother.jl")

Psmooth, loglike_smooth = bootstrap_smoother(rng, ys, init, fw_kernel, m_kernel, K)

Xsmooth = particles(Psmooth)

bf_output_smooth = marginalise(Psmooth, output_kernel)
Ysmooth = getindex.(mapreduce(permutedims, vcat, particles.(bf_output_filt)), 1)

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
        alpha = 0.05,
        label = "",
    )
end
display(state_smooth_plt)

output_smooth_plt = plot(ts, outs, label = "output", xlabel = "t", title = "log-variance")
plot!(ts, Ysmooth, color = "green", alpha = 0.01, label = "")
display(output_smooth_plt)
