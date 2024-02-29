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
AbstractDiracKernel
DiracKernel
IdentityKernel
```

#### Type aliases

```julia
const AffineDiracKernel{T} = DiracKernel{T,<:AbstractAffineMap}
```

### Constructors

```@docs
DiracKernel(Φ::AbstractMatrix)
DiracKernel(Φ::AbstractMatrix, b::AbstractVector)
DiracKernel(Φ::AbstractMatrix, b::AbstractVector, c::AbstractVector)
```

### Basics

```@docs
mean(K::DiracKernel)
```

### Conditioning and sampling

```@docs
condition(K::AbstractDiracKernel, x)
rand(::AbstractRNG, K::AbstractDiracKernel, x::AbstractVector)
rand(K::AbstractDiracKernel, x::AbstractVector)
```