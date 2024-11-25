# Flat likelihoods

```@meta
CurrentModule = MarkovKernels
```

A flat likelihood, $L$, acts as ideentity under Bayes' rule, that is:

```math
D(x) = \frac{L(x)D(x)}{\int L(x) D(x) dx}
```



### Type

```@docs
FlatLikelihood
```