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
```

### Methods

```@docs
dim(::Dirac)
mean(::Dirac)
```
