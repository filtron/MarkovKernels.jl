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
condition(K::AbstractStochasticMatrix, x)
rand(rng::AbstractRNG, K::AbstractStochasticMatrix, x::Int)
```

