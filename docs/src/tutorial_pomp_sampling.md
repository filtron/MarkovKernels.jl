# Sampling from Markov-realisable processes

This tutorial describes how to sample from the probabilistic state-space model given by

```math
\begin{aligned}
x_0 &\sim \mathcal{N}(\mu_0 ,\Sigma_0), \\
x_n \mid x_{n-1} &\sim \mathcal{N}(\Phi  x_{n-1}, Q),\\
y_n &= C x_n,
\end{aligned}
```

where $x$ and $y$ are referred to as the latent Gauss-Markov process and the output process, respectively.
Additionally, noisy measurements of the output process will be generated according to

```math
z_n \mid x_n \sim \mathcal{N}(Cx_n,R).
```


### Sampling from the latent Gauss-Markov process
```@example 1
using MarkovKernels, LinearAlgebra, Plots


N = 2^9
ns = 0:N

# define a Markov kernel for a homogeneous Markov proces
λ = 0.9
σ = 1.0
dimx = 2

Φ = [λ 0.0; 1 - λ^2 λ]
Q = (1-λ^2)*(1+λ^2) * 1.0*I(dimx)

forward_kernel = NormalKernel(Φ, Q)

# define initial distribution
init = Normal(zeros(dimx), 1.0*I(dimx))

# sample Gauss-Markov process and plot
xs = rand(init, forward_kernel, N)
plt_states = plot(
    ns,
    xs,
    layout=(dimx,1),
    xlabel = ["" "t"],
    label = ["x0" "x1"],
    title = ["Latent Gauss-Markov process" ""]
)
```

### Sampling output and measurements
```@example 1
# define output process
C = σ*[1.0 -1.0]
output_kernel = DiracKernel(C)

# define measurements of output process
R = fill(0.1,1,1)
measurement_kernel = NormalKernel(C,R)

# sample outputs, measurements and plot
output = rand(output_kernel,xs)
zs = rand(measurement_kernel,xs)

plt_output = plot(
    ns,
    output,
    xlabel = "t",
    ylabel = "y",
    label = "output process",
    title = "Output process and measurements"
)

scatter!(
    ns,
    zs,
    label = "measurements",
    color="black"
)
```