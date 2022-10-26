# DiracKernel

```@meta
CurrentModule = MarkovKernels
```

The Dirac kernel with conditional mean prameter  ``\mu`` is denotd by

```math
k(y\mid x) = \delta(y - \mu(x)).
```

### Types

```@docs
AbstractDiracKernel{T}
DiracKernel{T}
```

#### Type aliases

```julia
const AffineDiracKernel{T} = DiracKernel{T,<:AbstractAffineMap}
```

### Constructors

```@docs
DiracKernel(F::AbstractAffineMap)
DiracKernel(Φ::AbstractMatrix)
DiracKernel(Φ::AbstractMatrix, b::AbstractVector)
DiracKernel(Φ::AbstractMatrix, b::AbstractVector, c::AbstractVector)
DiracKernel{T}(K::AffineDiracKernel{U}) where {T,U}
```

### Basics

```@docs
mean(K::DiracKernel)
```

### Conditioning and sampling

```@docs
condition(K::DiracKernel, x)
rand(::AbstractRNG, K::AbstractDiracKernel, x::AbstractVector)
rand(K::AbstractDiracKernel, x::AbstractVector)
```