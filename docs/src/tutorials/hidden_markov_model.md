# Sampling and inference in Hidden Markov models


A finite state hidden Markov model is specified by an initial distribution, a sequence of transition probabilities,
and a sequence of observation probabilities:


```math
\begin{aligned}
&P(x_0 = i) \\
&P(x_t = i \mid x_{t-1} = j), \quad t = 1, \ldots, T   \\
&P(y_t = i \mid x_t = j), \quad t = 0, 1, \ldots, T
\end{aligned}
```

This tutorial describes how to use ```MarkovKernels.jl``` to:

* Sample from a (finite state) hidden Markov model
* Compute the a posteriori distribution of the hidden sequence using the backward recursion


```@example 1
using MarkovKernels
using Random, LinearAlgebra
import Plots
rng = Random.Xoshiro(19910215)

nothing # hide
```

## Implementing samplers for finite state (hidden) Markov models

```@example 1
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


## Defining a finite state hidden Markov model and sampling it


```@example 1
# number of possible hidden states and number of possible observation states
m, n = 10, 10

# probability vector of initial distribution
init = ProbabilityVector(ones(m))

# transition probabilities
Pxx = Matrix(Tridiagonal(ones(m - 1), 5 * ones(m), ones(m - 1)))
Kxx = StochasticMatrix(Pxx)

# observation probabilites
Pyx = (ones(m, m) - I)
Kyx = StochasticMatrix(Pyx)

T = 2^8 + 1
fw_kernels = fill(Kxx, T - 1)
obs_kernels = fill(Kyx, T)

# sample hidden and observed states
xs, ys = sample(rng, init, fw_kernels, obs_kernels)
nothing # hide
```

## Plotting the realization

```@example 1
hmm_plt = Plots.scatter(
    layout = (1, 2)
)
Plots.scatter!(
    hmm_plt,
    eachindex(xs),
    xs,
    color = "black",
    subplot = 1,
    title = "hidden states",
    label = "",
)
Plots.scatter!(
    hmm_plt,
    eachindex(ys),
    ys,
    color = "red",
    subplot = 2,
    title = "observation states",
    label = "",
)
```


## Implementing the backward algorithms

The backward algorith operates on the sequence of likelihoods of future observations:

```math
h_{t:T \mid s}(x) = P(y_{t:T} \mid x_s = x).
```
It computes an a posteriori initial distribution, a sequence of a posteriori transition probabilities, and a log-likelihood of the observations,
via a backward recursion.
The recursion is given by:

```math
\begin{aligned}
h_{t:T\mid t-1}(z) &= \sum_x h_{t:T \mid t}(x) P(x_t = x \mid x_{t-1} = z)  \\
P(x_t = x \mid x_{t-1} = z, y_{0:T}) &= h_{t:T \mid t}(x) P(x_t = x \mid x_{t-1} = z) / h_{t:T\mid t-1}(z) \\
h_{t-1:T \mid t-1}(x) &= h_{t:T\mid t-1}(x) h_{t-1 \mid t-1}(x)
\end{aligned}
```

The first two equations are implemented by ```htransform_and_likelihood``` and the last equation is implemented by ```compose```.
The algorithm terminates by computing the a posteriori initial distribution and the log-likelihood of the observations:

```math
\begin{aligned}
\log P(y_{0:T}) &=  \log \Big(\sum_x h_{0:T \mid 0}(x) P(x_0 = x) \Big) \\
P(x_0 = x \mid y_{0:T}) &= h_{0:T \mid 0}(x) P(x_0 = x) / P(y_{0:T})
\end{aligned}
```
These equations are implemented by ```posterior_and_loglike```.
Using ```MarkovKernels.jl```, the code might look something like the following:


```@example 1
function backward_recursion(init, forward_kernels, likelihoods)
    h = last(likelihoods)
    KT = Base.promote_op(first ∘ htransform_and_likelihood, eltype(forward_kernels), typeof(h))
    post_forward_kernels = Vector{KT}(undef, length(forward_kernels))

    for m in eachindex(forward_kernels)
        fw_kernel = forward_kernels[end-m+1]
        post_fw_kernel, h = htransform_and_likelihood(fw_kernel, h)
        post_forward_kernels[end-m+1] = post_fw_kernel

        like = likelihoods[end-m]
        h = compose(h, like)
    end
    post_init, loglike = posterior_and_loglike(init, h)
    return post_init, post_forward_kernels, loglike
end

nothing # hide
```


## Computing the a posteriori distribution of the hidden sequence


```@example 1
likes = [Likelihood(Kobs, y) for (Kobs, y) in zip(obs_kernels, ys)] # compute the likelihoods associated with the observations
post_init, post_fw_kernels, loglike = backward_recursion(init, fw_kernels, likes)

nothing # hide
```

## Sampling a posteriori hidden sequences and plotting them

```@example 1
nsample = 10
for _ in 1:nsample
    xs_post = sample(rng, post_init, post_fw_kernels)
    Plots.scatter!(
        hmm_plt,
        eachindex(xs_post),
        xs_post,
        label = "",
        color = "blue",
        alpha = 0.1,
        )
end
hmm_plt
```