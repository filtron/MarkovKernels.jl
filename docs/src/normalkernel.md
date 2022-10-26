# NormalKernel

```@meta
CurrentModule = MarkovKernels
```

The Normal kernel is denoted by

```math
k(y\mid x) = \mathcal{N}(y ; \mu(x)  , \Sigma(x) ).
```

As with the Normal distributions, the explicit expression on the kernel depends on whether it is real or complex valued.

### Types

```@docs
AbstractNormalKernel{T}
NormalKernel{T}
```

#### Type aliases

```julia
const AffineNormalKernel{T} = NormalKernel{T,<:AbstractAffineMap,<:CovarianceParameter}
```

### Constructors

```@docs
NormalKernel(F::AbstractAffineMap, Σ)
NormalKernel(Φ::AbstractMatrix, Σ)
NormalKernel(Φ::AbstractMatrix, b::AbstractVector, Σ)
NormalKernel(Φ::AbstractMatrix, b::AbstractVector, c::AbstractVector, Σ)
NormalKernel(F::AbstractAffineMap, Σ::CovarianceParameter)
NormalKernel(F::AbstractAffineMap, Σ::AbstractMatrix)
NormalKernel{T}(K::AffineNormalKernel{U}) where {T,U}
```

### Basics

```@docs
mean(K::NormalKernel)
cov(K::NormalKernel)
covp(K::NormalKernel)
```

### Conditioning and sampling

```@docs
condition(K::AbstractNormalKernel, x)
rand(RNG::AbstractRNG, K::AbstractNormalKernel, x::AbstractVector)
rand(K::AbstractNormalKernel, x::AbstractVector)
```
