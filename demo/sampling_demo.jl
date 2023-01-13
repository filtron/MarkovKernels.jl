using MarkovKernels
using Plots, Random, LinearAlgebra, IterTools

# sample a homogeneous Markov model
function sample(rng, init, K, nstep)
    x = rand(rng, init)
    it = Iterators.take(iterated(z -> rand(rng, K, z), x), nstep + 1)
    return mapreduce(permutedims, vcat, collect(it))
end

# set rng
rng = MersenneTwister(1991)

# time grid
m = 200
T = 5
ts = collect(LinRange(0, T, m))
dt = T / (m - 1)

# transtion kernel
λ = 2.0
Φ = exp(-λ * dt) .* [1.0 0.0; -2*λ*dt 1.0]
Q = I - exp(-2 * λ * dt) .* [1.0 -2*λ*dt; -2*λ*dt 1+(2*λ*dt)^2]
fw_kernel = NormalKernel(Φ, Q)

# initial distribution
init = Normal(zeros(2), 1.0I(2))

# sample state
xs = sample(rng, init, fw_kernel, m - 1)

state_plt = plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = "t",
    labels = ["x1" "x2"],
    title = ["Latent Gauss-Markov process" ""],
)
display(state_plt)

# output kernel and measurement kernel
C = 1.0 / sqrt(2) * [1.0 -1.0]
output_kernel = DiracKernel(C)
R = fill(0.1, 1, 1)
m_kernel = compose(NormalKernel(1.0I(1), R), output_kernel)

# sample output and its measurements
outs = mapreduce(z -> rand(rng, output_kernel, xs[z, :]), vcat, 1:m)
ys = mapreduce(z -> rand(rng, m_kernel, xs[z, :]), vcat, 1:m)

output_plot = plot(ts, outs, label = "output", xlabel = "t")
scatter!(ts, ys, label = "measurement", color = "black")
display(output_plot)
