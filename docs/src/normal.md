# Normal

```@meta
CurrentModule = MarkovKernels
```

The standard parametrisation of the Normal distribution is given by

```math
\mathcal{N}(x ; \mu  , \Sigma ),
```
where $\mu$ is the mean vector and $\Sigma$ is the covariance matrix.
The exact expression for the probabiltiy density function depends on whether $x$ is vector with real or complex values, both are supported.
For real valued vectors the density function is given by
```math
\mathcal{N}(x ; \mu  , \Sigma ) = |2\pi \Sigma|^{-1/2} \exp \Big(  -\frac{1}{2}(x-\mu)^* \Sigma^{-1} (x-\mu)  \Big),
```
whereas for complex valued vectors the density function is given by
```math
\mathcal{N}(x ; \mu  , \Sigma ) = |\pi \Sigma|^{-1} \exp \Big(  -(x-\mu)^* \Sigma^{-1} (x-\mu)  \Big).
```

* Types:

```julia
abstract type AbstractNormal{T<:Number}  <: AbstractDistribution end # normal distributions with realisations in real / complex Euclidean spaces
Normal{T} <: AbstractNormal{T} # mean vector / covariance matrix parametrisation of normal distributions
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
