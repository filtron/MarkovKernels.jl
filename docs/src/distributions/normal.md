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

### Types
```@docs
AbstractNormal{T}
Normal{T}
```

### Constructors

```@docs
Normal(μ::AbstractVector, Σ::CovarianceParameter)
Normal(μ::AbstractVector, Σ::AbstractMatrix)
Normal{T}(N::Normal{U,V,W}) where {T,U,V<:AbstractVector,W<:CovarianceParameter}
```

### Basics

```@docs
dim(::Normal)
mean(::Normal)
cov(::Normal)
covp(::Normal)
var(::AbstractNormal)
std(::AbstractNormal)
```

### Probability density function

```@docs
residual(N::AbstractNormal, x::AbstractVector)
logpdf(N::AbstractNormal{T}, x) where {T}
```

### Information theory

```@docs
entropy(::AbstractNormal)
kldivergence(N1::AbstractNormal{T}, N2::AbstractNormal{T}) where {T<:Number}
```

### Sampling

```@docs
rand(rng::AbstractRNG, N::AbstractNormal) 
```
