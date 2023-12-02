# General 

```@meta
CurrentModule = MarkovKernels
```

### Type information 

```@docs
typeof_sample(D::AbstractDistribution)
eltype_sample(D::AbstractDistribution)
```

### Probability densities

```@docs 
logpdf(D::AbstractDistribution, x) 
```

### Sampling 

```@docs 
Base.rand(rng::AbstractRNG, D::AbstractDistribution)
```