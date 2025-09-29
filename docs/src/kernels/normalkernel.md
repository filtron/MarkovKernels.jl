# NormalKernel

```@meta
CurrentModule = MarkovKernels
```

The Normal kernel is denoted by

```math
k(y\mid x) = \mathcal{N}(y ; \mu(x)  , \Sigma(x) ).
```

As with the Normal distributions, the explicit expression on the kernel depends on whether it is real or complex valued.

### Types

```@docs
AbstractNormalKernel
NormalKernel
```

#### Type aliases

```julia
const HomoskedasticNormalKernel{TM,TC} = NormalKernel{<:Homoskedastic,TM,TC} where {TM,TC} # constant conditional covariance
const AffineHomoskedasticNormalKernel{TM,TC} =
    NormalKernel{<:Homoskedastic,TM,TC} where {TM<:AbstractAffineMap,TC} # affine conditional mean, constant conditional covariance
const AffineHeteroskedasticNormalKernel{TM,TC} =
    NormalKernel{<:Heteroskedastic,TM,TC} where {TM<:AbstractAffineMap,TC} # affine conditional mean, non-constant covariance
const NonlinearNormalKernel{TM,TC} = NormalKernel{<:Heteroskedastic,TM,TC} where {TM,TC} # the general, nonlinear case
```

### Constructors

```@docs
NormalKernel(μ, Σ)
```

### Methods

```@docs
mean(K::NormalKernel)
cov(K::NormalKernel)
covparam(K::NormalKernel)
rand(rng::AbstractRNG, K::AbstractNormalKernel, x::AbstractNumOrVec)
```
