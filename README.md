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
Probabilistic state-space model:

$$ 
\displaylines{x_n \mid x_{n-1} \sim f_n(\cdotp\mid x_{n-1}) \\
y_n \mid x_n \sim m_n(\cdotp\mid x_n)}
$$


* Bayesian filtering:
```julia
f_dists, p_dists, bw_kernels, loglike = filter(init,fw_kernels,likelihoods,aligned)
```

* Bayesian smoothing: 
```julia
s_dists, f_dists, p_dists, bw_kernels, loglike = smoother(init,fw_kernels,likelihoods,aligned)
```

