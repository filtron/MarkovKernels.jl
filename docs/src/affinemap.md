# asd

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
AbstractAffineMap{T}(F::AffineMap)
```

### Basics

```@docs
slope(F::AffineMap)
intercept(F::AffineMap)
compose(F2::AbstractAffineMap, F1::AbstractAffineMap)
*(F2::AbstractAffineMap, F1::AbstractAffineMap)
```