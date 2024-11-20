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
Normal(μ::AbstractVector{<:Real}, Σ::Symmetric{<:Real})
Normal(μ::AbstractVector{<:Complex}, Σ::Hermitian{<:Complex})
Normal(μ::AbstractVector, Σ::Cholesky)
Normal(μ::Number, Σ::Real)
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
logpdf(N::AbstractNormal, x)
```

### Information theory

```@docs
entropy(::AbstractNormal)
kldivergence(N1::AbstractNormal, N2::AbstractNormal)
```

### Sampling

```@docs
rand(rng::AbstractRNG, N::AbstractNormal) 
```
