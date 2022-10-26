# MarkovKernels.jl 

A package implementing distributions, Markov kernels, and likelihoods that all play nice with eachother. 
The main motivation is to simplify the implementation of Bayesian filtering and smoothing algorithms. 
Let $\pi(x)$ be a probability distribution and $k(y\mid x)$ a Markov kernel then only the following operations are required for Bayesian state estimation

* Marginalisation: 

$$
k(y) = \int k(y\mid x) \pi(x) \mathrm{d} x, 
$$ 

which gives the prediction step in Bayesian filtering. 

* Inverse factorisation: 

$$
k(y\mid x)\pi(x) = \pi(x \mid y) k(y),  
$$

where evaluation of $\pi(x \mid y)$ at $y$ gives Bayes' rule and $k(y)$ is the marginal distribution of $y$ (used for prediction error decomposition of the marginal likelihood). In fact, the prediction step may be implemented with the inverse factorisation operation as well, in which case $\pi(x\mid y)$ is the backwards kernel used to compute smoothing distributions in the Rauch-Tung-Striebel recursion. 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://filtron.github.io/MarkovKernels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://filtron.github.io/MarkovKernels.jl/dev/)
[![Build Status](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/filtron/MarkovKernels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/filtron/MarkovKernels.jl)

## Installation 

```julia 
] add https://github.com/filtron/MarkovKernels.jl.git
```

## Package specific types

Types for representing marginal distributions, Markov kernels, and likelihoods:

```julia
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


For the purpose of Bayesian state estimation, ideally the following functions are defined:   

```julia
marginalise(D::AbstractDistrbution, K::AbstractMarkovKernel)
invert(D::AbstractDistrbution, K::AbstractMarkovKernel)
bayes_rule(D::AbstractDistrbution, K::AbstractMarkovKernel)
```

However, these can rarely be implemented exactly for arbitrary distribution / Markov kernel pars.
Therefore, it is in practice up to the user to define appropriate approximations, i.e. 

```julia
predict(D::AbstractDistribution,K::AbstractMarkovKernel)
update(D::AbstractDistribution,L::AbstractLikelihood)
```

