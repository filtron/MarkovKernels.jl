# Normal

```@meta
CurrentModule = MarkovKernels
```

* The normal distribution is denoted by

```math
\pi(x) = \mathcal{N}(x ; \mu  , \Sigma ).
```

The exact expression for the probabiltiy density function depends on whether $x$ is vector with real or complex values, both are supported.

* Types:

```julia
abstract type AbstractNormal{T<:Number}  <: AbstractDistribution end # normal distributions with realisations in real / complex Euclidean spaces
Normal{T} <: AbstractNormal{T} # mean vector / covariance matrix parametrisation of normal distributions
Dirac{T}  <: AbstractNormal{T} # normal distribution with zero covariance
```

* Functionality:
