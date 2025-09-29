# Probability vector

## Types

```@docs
AbstractProbabilityVector{T}
ProbabilityVector{T}
```

## Constructor
```@docs
ProbabilityVector(p::AbstractVector)
```


## Methods
```@docs
probability_vector(::AbstractProbabilityVector)
entropy(C::AbstractProbabilityVector)
kldivergence(C1::AbstractProbabilityVector, C2::AbstractProbabilityVector)
```


