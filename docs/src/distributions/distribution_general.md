# General

```@meta
CurrentModule = MarkovKernels
```

### Type

```@docs
AbstractDistribution{T<:Number}
```


### Type information

```@docs
sample_type(D::AbstractDistribution)
sample_eltype(D::AbstractDistribution)
```

### Probability densities

```@docs
logpdf(D::AbstractDistribution, x)
```

### Sampling

```@docs
Base.rand(rng::AbstractRNG, D::AbstractDistribution)
```