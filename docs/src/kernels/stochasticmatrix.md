# StochasticMatrix

```@meta
CurrentModule = MarkovKernels
```


### Types

```@docs
AbstractStochasticMatrix
StochasticMatrix
```

## Constructor

```@docs
StochasticMatrix(P::AbstractMatrix)
```


## Methods

```@docs
probability_matrix(::AbstractStochasticMatrix)
rand(rng::AbstractRNG, K::AbstractStochasticMatrix, x::Int)
```

