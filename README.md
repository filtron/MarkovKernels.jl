# MarkovKernels.jl 

A package implementing distributions, Markov kernels, and likelihoods that all play nice with eachother. 
The main motivation is to simplify the implementation of Bayesian filtering and smoothing algorithms. 
Let $\pi(x)$ be a probability distribution and $k(y\mid x)$ a Markov kernel then only the following operations are required for Bayesian state estimation

* Marginalization: 

$$
k(y) = \int k(y\mid x) \pi(x) \mathrm{d} x, 
$$ 

which gives the prediction step in Bayesian filtering. 

* Inverse factorization: 

$$
k(y\mid x)\pi(x) = \pi(x \mid y) k(y),  
$$

where evaluation of $\pi(x \mid y)$ at $y$ gives Bayes' rule and $k(y)$ is the marginal distribution of $y$ (used for prediction error decomposition of the marginal likelihood). In fact, the prediction step may be implemented with the inverse factorization operation as well, in which case $\pi(x\mid y)$ is the backwards kernel used to compute smoothing distributions in the Rauch-Tung-Striebel recursion. 
Please see the tutorials in the documentation. 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://filtron.github.io/MarkovKernels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://filtron.github.io/MarkovKernels.jl/dev/)
[![Build Status](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/filtron/MarkovKernels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/filtron/MarkovKernels.jl)

## Installation 

```julia 
] add MarkovKernels
```

## Package specific types

Types for representing marginal distributions, Markov kernels, and likelihoods:

```julia
abstract type AbstractAffineMap end # used to represent affine conditional means

abstract type AbstractDistribution end
abstract type AbstractMarkovKernel end
abstract type AbstractLikelihood end
```

Currently, the following concrete types are defined: 

```julia
Normal # Vector valued Normal distributons 
Dirac  # Vector valued Dirac distributions 

NormalKernel # Vector valued Normal kernels 
DiracKernel  # Vector valued Dirac kernels 

Likelihood   # AbstractMarkovKernel paired with a measurement 
```

The following type union is used to represent the (conditional) covariance:

```julia 
const CovarianceParameter{T} = Union{HermOrSym{T},Factorization{T}}
```

Additionally, the following aliases are defined: 

```julia 
const AffineNormalKernel{T} = NormalKernel{T,<:AbstractAffineMap,<:CovarianceParameter}
const AffineDiracKernel{T} = DiracKernel{T,<:AbstractAffineMap}
```

## Functions 

For the purpose of Bayesian state estimation, ideally the following functions are defined:   

```julia
marginalize(D::AbstractDistribution, K::AbstractMarkovKernel)
invert(D::AbstractDistribution, K::AbstractMarkovKernel)
bayes_rule(D::AbstractDistribution, L::AbstractLikelihood)
```

These are currently implemented for Normal, AffineNormalKernel, AffineDiracKernel. 
Additionally, marginalize is implemented for Dirac with respect to the aforementioned kernels. 

In practice, these functions can not be implemented exactly for a given distribution / Markov kernel pair.
Therefore, it is up to the user to define, when required, appropriate approximations, i.e.: 

```julia
predict(D::AbstractDistribution, K::AbstractMarkovKernel)
update(D::AbstractDistribution, L::AbstractLikelihood)
```
