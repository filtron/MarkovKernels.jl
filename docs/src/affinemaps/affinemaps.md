# Affine maps

An affine map is a function ``f`` given by

```math
 f(x) = A x + b,
```

where ``A`` is the slope and ``b`` is the intercept.
Different representations of affine maps are sometimes useful, as documented below.


```@meta
CurrentModule = MarkovKernels
```

### Types

```@docs
AbstractAffineMap{T<:Number}
AffineMap{T,U,V}
LinearMap{T,U}
AffineCorrector{T,U,V,S}
```

### Constructors

```@docs
AffineMap(A::AbstractMatrix, b::AbstractVector)
LinearMap(A::AbstractMatrix)
AffineCorrector(A::AbstractMatrix, b::AbstractVector, c::AbstractVector)
```

### Basics

```@docs
slope(F::AffineMap)
intercept(F::AffineMap)
compose(F2::AbstractAffineMap, F1::AbstractAffineMap)
∘(F2::AbstractAffineMap, F1::AbstractAffineMap)
```