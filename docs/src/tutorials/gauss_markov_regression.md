# Sampling and inference in Gauss-Markov models

Gauss-Markov realizable signal:

```math
\begin{aligned}
x_0 &\sim \mathcal{N}(\mu_0, \Sigma_0) \\
x_t \mid x_u &\sim \mathcal{N}( \Phi_{t, u} x_u, Q_{t, u}) \\
s_t \mid x_t &\sim \delta(\cdotp - C x_t)
\end{aligned}
```
Transition parameters:
```math
\begin{aligned}
\Phi_{t, u} &= e^{A (t - u)}\\
Q_{t, u} &= \sqrt{t-u} \int_0^1 e^{A(t-u)z} B B^* e^{A^*(t-u)z} \mathrm{d} z
\end{aligned}
```

Observations:

```math
y_{t_k} \mid s_{t_k} \sim \mathcal{N}(s_{t_k}, R)
```



```@example 2
using MarkovKernels
using FiniteHorizonGramians
using Random, LinearAlgebra
import Plots
rng = Random.Xoshiro(19910215)

nothing # hide
```

## Implementing samplers for Gauss-Markov realizable signals

The code for sampling from a Gauss-Markov realizable signal is exactly the same as in the hidden Markov model tutorial. 
For completeness, the code is given below. 


```@example 2
function sample(rng::AbstractRNG, init, fw_kernels)
    x = rand(rng, init)
    n = length(fw_kernels) + 1
    xs = Vector{typeof(x)}(undef, n)
    xs[begin] = x

    for (m, fw_kernel) in pairs(fw_kernels)
        x = rand(rng, condition(fw_kernel, x))
        xs[begin+m] = x
    end
    return xs
end

function sample(rng::AbstractRNG, init, fw_kernels, obs_kernels)
    # sample initial values
    x = rand(rng, init)
    y = rand(rng, first(obs_kernels), x)

    # allocate output
    n = length(obs_kernels)
    xs = Vector{typeof(x)}(undef, n)
    ys = Vector{typeof(y)}(undef, n)

    xs[begin] = x
    ys[begin] = y

    for (m, fw_kernel) in pairs(fw_kernels)
        obs_kernel = obs_kernels[begin+m]
        x = rand(rng, condition(fw_kernel, x))
        y = rand(rng, condition(obs_kernel, x))
        xs[begin+m] = x
        ys[begin+m] = y
    end
    return xs, ys
end

nothing # hide
```


## Defining a Gauss-Markov realizable signal and computing transition kernels 

Implementation of transition kernel computation for continuous-time Gauss-Markov processes:

```@example 2
function transition_kernel(A, B, dt)
    alg = FiniteHorizonGramians.ExpAndGram{Float64,13}()
    Φ, U = FiniteHorizonGramians.exp_and_gram_chol(A, B, dt, alg)
    Φ = LinearMap(Φ)
    Q = Cholesky(UpperTriangular(U))
    return NormalKernel(Φ, Q)
end

nothing # hide
```

Definiiton of continuous-time Gauss-Markov realizable process: 

```@example 2
function gauss_markov_realizable_model(λ, σ, p)
A = λ * (I - 2 * tril(ones(p, p)))
B = sqrt(2λ) * ones(p, 1)
Cadj =
    σ * factorial(p - 1) / sqrt(factorial(2(p - 1))) .* binomial.(p - 1, 0:p-1) .*
    (-1) .^ (0:p-1)
C = adjoint(Cadj)
return A, B, C
end

nothing # hide
```

Computing initial density, transition kernels, output kernel, and observation kernels: 

```@example 2
# time grid 
n = 2^9 + 1
T = 20.0
ts = LinRange(zero(T), T, n)

# model parameters
λ = 2.0
σ = 1.0
p = 3

# model matrices 
A, B, C = gauss_markov_realizable_model(λ, σ, p)


init = Normal(zeros(p), cholesky(diagm(0 => ones(p))))
forward_kernels = [transition_kernel(A, B, dt) for dt in diff(ts)]
output_kernel = DiracKernel(LinearMap(C))
output_kernels = fill(output_kernel, n)

nothing # hide
```

Simulating the latent state and the signal: 
```@example 2
xs, ss = sample(rng, init, forward_kernels, output_kernels)

nothing # hide
```

Defining observation kernel and simulating observations: 
```@example 2
R = 0.1 
observation_kernel = NormalKernel(LinearMap(one(R)), R)
ys = [rand(rng, condition(observation_kernel, s)) for s in ss]

obs_idx = [mod(n, 2^3) == 1 for n in eachindex(ts)]
obs_ts = ts[obs_idx]
ys = Union{eltype(ys), Missing}[obs_idx[n] ? ys[n] : missing for n in eachindex(obs_idx)]

nothing # hide
```



## Plotting the realization


```@example 2
gmr_plt = Plots.plot(
    ts, 
    ss, 
    color = "black",
    label = "",
    xlabel = "t"
)
Plots.scatter!(
    gmr_plt, 
    obs_ts, 
    filter(!ismissing, ys), 
    color = "red", 
    label = ""
)
gmr_plt
```

## Implementing the forward algorithm

The forward algorithm operates on a sequence of a posteriori terminal distributions (so-called filtering distributions):
```math
p(x_t \mid y_{0:t})
```
The algorithm first initializes by:
```math
\begin{aligned}
p(y_0) &= \int p(y_0 \mid x_0) p(x_0) \mathrm{d} x_0 \\
p(x_0 \mid y_0) &= p(y_0 \mid x_0) p(x_0)
\end{aligned}
```
These two equations are implemented by ```posterior_and_loglike```.
The algorithm then computes a sequence of filtering distributions, reverse-time transition kernels, and accumulates the log-likelihood of the observations according to
the following forward recursion:
```math
\begin{aligned}
p(x_t \mid y_{0:t-1}) &= \int p(x_t \mid x_{t-1}) p(x_{t-1} \mid y_{0:t-1}) \mathrm{d} x_{t-1} \\
p(x_{t-1} \mid x_t, y_{0:T}) &= p(x_t \mid x_{t-1}) p(x_{t-1} \mid y_{0:t-1}) / p(x_t \mid y_{0:t-1}) \\
p(y_t \mid y_{0:t-1}) &= \int p(y_t \mid x_t) p(x_t \mid y_{0:t-1}) \mathrm{d} x_t \\
\log p(y_{0:t}) &= \log p(y_{0:t-1}) + \log p(y_t \mid y_{0:t-1})
\end{aligned}
```
The first two equations are implemented by ```invert``` and the last two equations are, again, implemented by ```posterior_and_loglike```
Using ```MarkovKernels.jl```, the code might look something like the following:


```@example 2
function forward_recursion(init, forward_kernels, likelihoods)
    like = first(likelihoods)
    filt, loglike = posterior_and_loglike(init, like)
    KT = Base.promote_op(last ∘ invert, typeof(filt), eltype(forward_kernels))
    backward_kernels = Vector{KT}(undef, length(forward_kernels))
    for m in eachindex(forward_kernels)
        fw_kernel = forward_kernels[m]
        pred, bw_kernel = invert(filt, fw_kernel)

        like = likelihoods[m+1]
        filt, ll_incr = posterior_and_loglike(pred, like)
        loglike = loglike + ll_incr
        backward_kernels[m] = bw_kernel
    end
    return backward_kernels, filt, loglike
end

nothing # hide
```

## Computing a reverse-time Markov a posteriori path-distribution

Compute likelihoods: 
```@example 2
likelihoods = [Likelihood(compose(observation_kernel, output_kernel), y) for y in ys]

nothing # hide
```

Compute reverse-time posterior: 
```@example 2
backward_kernels, term, loglike = forward_recursion(init, forward_kernels, likelihoods)

nothing # hide
```



## Computing a posteriori time-marginals
The a posteriori time marginals may be computed according to the following backward recursion:
```math
p(x_{t-1} \mid y_{0:T}) = \int p(x_{t-1} \mid x_t, y_{0:T}) p(x_t \mid y_{0:T}) \mathrm{d} x_t
```
This equation is implemented by ```marginalize```.
Using ```MarkovKernels.jl```, the code might look something like the following:

```@example 2
function reverse_time_marginals(term, bw_kernels)
    dist = term 
    dists = Vector{typeof(init)}(undef, length(bw_kernels)+1)
    dists[end] = dist 
    for (m, kernel) in pairs(reverse(bw_kernels))
        dist = marginalize(dist, kernel)
        dists[end-m] = dist
    end
    return dists
end
```

The time-marginals are then computed like so: 
```@example 2
post_state_dists = reverse_time_marginals(term, backward_kernels)
post_output_dists = [marginalize(dist, output_kernel) for dist in post_state_dists]

nothing # hide
```

## Plotting the a posteriori time marginals and some samples

```@example 2
Plots.plot!(
    gmr_plt, 
    ts, 
    post_output_dists,
    color = "blue",
    label = "",
)

nsample = 5
for _ in 1:nsample 
    _, ss_post = sample(rng, term, reverse(backward_kernels), output_kernels)
    ss_post = reverse(ss_post)
    Plots.plot!(
        gmr_plt,
        ts, 
        ss_post, 
        color = "blue",
        label = "", 
    )
end
gmr_plt
```