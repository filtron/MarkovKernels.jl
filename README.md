# MarkovKernels.jl 

A package implementing defining a Distributions, Markov kernels, and likelihoods that all play nice with eachother. 
The main motivation is to simplify the implementation of Bayesian filtering and smoothing algorithms. 

[![Build Status](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/filtron/MarkovKernels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/filtron/MarkovKernels.jl)

## Package specific types

* Types for representing marginal distributions, Markov kernels, and likelihoods:

```julia
abstract type AbstractDistribution end
abstract type AbstractMarkovKernel end
abstract type AbstractLikelihood end
```
### Normal distributions 
Normal distributions are implemented: 

```julia
abstract type AbstractNormal end 
Normal <: AbstractNormal # mean vector / covariance matrix parametrisation of normal distributions 
Dirac  <: AbstractNormal # normal distribution with zero covariance 
```

The following functions are provided:  

```julia
dim(N::AbstractNormal) 

mean(N::AbstractNormal) 
cov(N::AbstractNormal) 
var(N::AbstractNormal) 
std(N::AbstractNormal) 

residual(N::AbstractNormal,x) 
logpdf(N::AbstractNormal,x)
entropy(N::AbstractNormal)
kldivergence(N1::AbstractNormal,N2::AbstractNormal) 
rand(N::AbstractNormal)
```


