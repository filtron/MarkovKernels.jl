# MarkovKernels.jl 

A package implementing Bayesian filtering and smoothing by manipulating marginal distributions and Markov kernels.

[![Build Status](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/filtron/MarkovKernels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/filtron/MarkovKernels.jl)

## Package specific types

* Types for representing marginal distributions, Markov kernels, and likelihoods:

```julia
abstract type AbstractDistribution end
abstract type AbstractMarkovKernel end
abstract type AbstractLikelihood end
```

For Bayesian state estimation, the following two methods need to be defined:

```julia
d_pred, bw_kernel = predict(d::AbstractDistribution,k::AbstractMarkovKernel)

d_new, prediction_error_distribution, loglike = update(d::AbstractDistribution,l::AbstractLikelihood)
```



* Type for representing state-estimation problems: 

```julia
abstract type AbstractStateEstimationProblem end
```


## Bayesian state estimation 
A probabilistic state-space model may be defined by a family of forward kernel, $f_n$, and a family of measurement kernels $m_n$ according to 

$$ 
\displaylines{x_n \mid x_{n-1} \sim f_n(\cdotp\mid x_{n-1}), \\
y_n \mid x_n \sim m_n(\cdotp\mid x_n),}
$$

where $x_n$ is the state sequence, $y_n$ is the measurement sequence, and for fixed $y_n$ the map $x_n \mapsto m_n(y_n\mid x_n)$ is a likelihood. 
Bayesian state estimation problems consist of computing marginal conditional distributions of the state given some measurement sequence $y_{1:N}$. For the state-space model to be well posed, an initial condition for the state sequence is required. 
If it is given by $x_1 \sim \pi(x_1)$ the problem is said to be aligned, and non-aligned if it is given by $x_0 \sim \pi(x_0)$. 

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
The filtering problem is mathematically solved via the following prediction / update recursion. 

* Predict: 

$$
\displaylines{ \pi(x_n \mid y_{1:n-1}) = \int f_n(x_n\mid x_{n-1}) \pi(x_{n-1}\mid y_{1:n-1}) \mathrm{d} x_{n-1}, \\
\beta(x_{n-1} \mid x_n,y_{1:n-1}) \propto  f_n(x_n\mid x_{n-1}) \pi(x_{n-1}\mid y_{1:n-1}).
}
$$

* Update: 

$$
\displaylines{ \pi(x_n \mid y_{1:n}) \propto \pi(x_n \mid y_{1:n-1}) m_n(y_n\mid x_n), \\
p(y_n \mid y_{1:n-1}) = \int m_n(y_n\mid x_n) \pi(x_n\mid y_{1:n-1}) \mathrm{d} x_n .
}
$$

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
The smoothing problem is mathematically solved via the following backward recursion. 

$$
\gamma(x_n \mid y_{1:N}) = \int \beta(x_n \mid x_{n+1}, y_{1:n}) \gamma(x_{n+1} \mid y_{1:N}) \mathrm{d} x_{n+1}
$$

In code:
```julia
smooth_dists, filter_dists, pred_dists, bw_kernels, loglike = smoother(init,fw_kernels,likelihoods)
```

