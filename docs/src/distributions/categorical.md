# Categorical

## Types

```@docs
AbstractCategorical{T}
Categorical{T}
```

## Constructor
```@docs
Categorical(p::AbstractVector)
```


## Methods
```@docs
probability_vector(::AbstractCategorical)
entropy(C::AbstractCategorical)
kldivergence(C1::AbstractCategorical, C2::AbstractCategorical)
```


