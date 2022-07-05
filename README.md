# MarkovKernels.jl 

A package implementing Bayesian filtering and smoothing by manipulating marginal distributions and Markov kernels.

[![Build Status](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/filtron/MarkovKernels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/filtron/MarkovKernels.jl)

## Package specific types

* Type for representing marginal distributions: 

```julia
abstract type AbstractDistribution end
```

* Type for representing Markov kernels: 

```julia
abstract type AbstractMarkovKernel end
```

* Type for representing likelihoods: 

```julia
abstract type AbstractLikelihood end
```

## Bayesian state estimation 
A probabilistic state-space model may be defined by a family of forward kernel, $f_n$, and a family of measurement kernels $m_n$ according to 

$$ 
\displaylines{x_n \mid x_{n-1} \sim f_n(\cdotp\mid x_{n-1}), \\
y_n \mid x_n \sim m_n(\cdotp\mid x_n)},
$$

where $x_n$ is the state sequence, $y_n$ is the measurement sequence, and for fixed $y_n$ the map $x_n \mapsto m_n(y_n\mid x_n)$ is a likelihood. 
Bayesian state estimation problems consist of computing marginal conditional distributions of the state given some measurement sequence $y_{1:N}$. For the state-space model to be well posed, an initial condition for the state sequence is required. 
If it is given by $x_1 \sim \pi(x_1)$ the problem is said to be aligned, and non-aligned if it is given by $x_0 \sim \pi(x_0)$. 
Other possible initial conditions are currently not supported (but can be done with some fiddling?). 

### The filtering problem 
The filter problem consists of computing the following conditional distributions

$$
\displaylines{ \pi(x_n \mid y_{1:n}), \\ 
p(y_n \mid y_{1:n-1}), \\
\beta(x_n \mid x_{n+1},y_{1:n}), } 
$$

where $\pi$ denotes the filter distributions.
The one-step ahead measurement prediction distributions $p$ can be used to compute log-likelihoods,
and the backward kernels $\beta$ can be used to solve the smoothing problem.  

In code:
```julia 
filter_dists, pred_dists, bw_kernels, loglike = filter(init,fw_kernels,likelihoods) 
```
### The smoothing problem 
The smoothing problem consists of computing the following 

$$
\displaylines{ \gamma(x_n \mid y_{1:N}), \\
\pi(x_n \mid y_{1:n}), \\ 
p(y_n \mid y_{1:n-1}), \\
\beta(x_n \mid x_{n+1},y_{1:n}), } 
$$

where $\gamma$ denotes the smoothing distributions, and all other quantities have previously been defined in context of the filtering problem. 

In code:
```julia
smooth_dists, filter_dists, pred_dists, bw_kernels, loglike = smoother(init,fw_kernels,likelihoods)
```

