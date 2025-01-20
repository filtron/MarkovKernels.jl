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
using Random, LinearAlgebra, Plots
rng = Random.Xoshiro(19910215)

nothing # hide
```

## Implementing samplers for Gauss-Markov realizable signals

```@example 2

nothing # hide
```


## Defining a Gauss-Markov realizable signal and sampling it


```@example 2

nothing # hide
```

## Plotting the realization


```@example 2

nothing # hide
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
The first two equations are implemented by ```invert``` and the last two equations are, again, implemented by ```posterior_and_loglike```.
Using ```MarkovKernels.jl```, the code might look something like the following:


```@example 2

nothing # hide
```

## Computing a reverse-time Markov a posteriori path-distribution


```@example 2

nothing # hide
```

## Computing a posteriori time-marginals
The a posteriori time marginals may be computed according to the following backward recursion:
```math
p(x_{t-1} \mid y_{0:T}) = \int p(x_{t-1} \mid x_t, y_{0:T}) p(x_t \mid y_{0:T}) \mathrm{d} x_t
```
This equation is implemented by ```marginalize```.
Using ```MarkovKernels.jl```, the code might look something like the following:


## Plotting the a posteriori time marginals and some samples