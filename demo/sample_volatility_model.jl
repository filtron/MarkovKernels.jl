using MarkovKernels
using Plots, Random
using IterTools

rng = MersenneTwister(1991)

include("sampling_implementation.jl")
include("pf_implementation.jl")
# time grid
m = 100
T = 10
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

P = 5000

particles = particle_filter(rng, ys, init, fw_kernel, m_kernel, P)

aval = 0.0075

state_plt = plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = "t",
    labels = ["x1" "x2"],
    title = ["Latent Gauss-Markov process" ""],
)
for p in 1:P
    plot!(ts, components(particles)[p], color = "black", alpha = aval, label = "")
end
display(state_plt)

output_plot = plot(ts, outs, label = "output", xlabel = "t", title = "log-variance")
for p in 1:P
    plot!(
        ts,
        mapreduce(permutedims, vcat, mean(components(particles)[p])) * C',
        color = "black",
        alpha = aval,
        label = "",
    )
end
display(output_plot)
