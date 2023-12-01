# Likelihood

```@meta
CurrentModule = MarkovKernels
```


### Types

```@docs
FlatLikelihood
Likelihood{U,V}
```
### Constructors

```@docs
Likelihood(K::AbstractMarkovKernel, y)
```

### Basics

```@docs
measurement_model(L::Likelihood)
measurement(L::Likelihood)
log(L::Likelihood, x) 
```

