# Sampling from Markov-realisable processes

This tutorial describes how to sample from the probabilistic state-space model given by

```math
\begin{aligned}
x_0 &\sim \mathcal{N}(\mu_0 ,\Sigma_0), \\
x_n \mid x_{n-1} &\sim \mathcal{N}(\Phi  x_{n-1}, Q),\\
z_n &= C x_n,
\end{aligned}
```

where $x$ and $z$ are referred to as the latent Gauss-Markov process and the output process, respectively.
Additionally, noisy measurements of the output process will be generated according to

```math
y_n \mid x_n \sim \mathcal{N}(Cx_n,R).
```

## Sampling a Gauss-Markov process
```@example 1
using MarkovKernels
using Random, LinearAlgebra, Plots, IterTools

# sample a homogeneous Markov model
function sample(rng, init, K, nstep)
    x = rand(rng, init)
    it = Iterators.take(iterated(z -> rand(rng, K, z), x), nstep + 1)
    return mapreduce(permutedims, vcat, collect(it))
end
```

## Sampling latent states
```@example 1
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

plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = "t",
    labels = ["x1" "x2"],
    title = ["Latent Gauss-Markov process" ""],
)
```

## Sampling and plotting the output

```@example 1
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
```
