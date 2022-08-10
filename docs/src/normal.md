# Normal

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