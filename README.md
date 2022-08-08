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

* Type

```julia
abstract type AbstractNormal{T<:Number}  <: AbstractDistribution end # normal distributions with realisations in real / complex Euclidean spaces  
Normal{T} <: AbstractNormal{T} # mean vector / covariance matrix parametrisation of normal distributions 
Dirac{T}  <: AbstractNormal{T} # normal distribution with zero covariance 
```

The following functions are provided:  

```julia
dim(N::AbstractNormal)  # dimension  of the normal distribution 

mean(N::AbstractNormal) # mean vector 
cov(N::AbstractNormal)  # covariance matrix 
var(N::AbstractNormal)  # vector of marginal variances 
std(N::AbstractNormal)  # vector of marginal standard deviations 

residual(N::AbstractNormal,x) # whitened residual of realisation x
logpdf(N::AbstractNormal,x)   # logarithm of the probability density function at x 
entropy(N::AbstractNormal)   
kldivergence(N1::AbstractNormal,N2::AbstractNormal) 
rand(N::AbstractNormal) 
```

### Normal kernels 

```julia
abstract type AbstractNormalKernel{T<:Number}  <: AbstractMarkovKernel end # normal kernel over real / complex Euclidean spaces  
NormalKernel{T} <:  AbstractNormalKernel{T}  # normal kernels with mean function / homoscedastic covariance 
DiracKernel{T}  <:  AbstractNormalKernel{T}  # same as above buit with zero covariance 
```

