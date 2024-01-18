# Dirac

```@meta
CurrentModule = MarkovKernels
```

The Dirac distribution with parameter ``\mu`` is a distribution putting all probabiltiy mass on ``\mu``.
It is denoted by
```math
\delta(x -\mu).
```

### Types
```@docs
AbstractDirac{T}
Dirac{T}
```

### Constructors
```@docs
Dirac(μ::AbstractVector)
Dirac{T}(D::Dirac{U,V}) where {T,U,V<:AbstractVector}
```

### Basics

```@docs
dim(::Dirac)
mean(::Dirac)
```

### Sampling

```@docs
rand(::AbstractRNG, ::AbstractDirac)
```
