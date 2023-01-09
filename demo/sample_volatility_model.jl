using MarkovKernels
using Plots, Random
using IterTools

rng = MersenneTwister(1991)

include("sampling_implementation.jl")
include("bootstrap_filter.jl")

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

K = 200

Ps, loglike = bootstrap_filter(rng, ys, init, fw_kernel, m_kernel, K)

X = mapreduce(permutedims, vcat, particles.(Ps))

bf_output = [marginalise(Ps[i], output_kernel) for i in eachindex(Ps)]
Y = getindex.(mapreduce(permutedims, vcat, particles.(bf_output)), 1)

state_plt = plot(
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
        mapreduce(permutedims, vcat, X[:, k]),
        markersize = 1,
        color = "red",
        alpha = 0.05,
        label = "",
    )
end
display(state_plt)

output_plot = plot(ts, outs, label = "output", xlabel = "t", title = "log-variance")
scatter!(ts, Y, markersize = 1, color = "red", alpha = 0.05, label = "")

display(output_plot)

#scatter!(ts, bf_output, color = "red", alpha = 0.0025)
