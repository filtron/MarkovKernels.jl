# Covariance Parameter

```@meta
CurrentModule = MarkovKernels
```

In MarkovKernels.jl a covariance matrix is assumed to be of the following type union.

```julia
const CovarianceParameter{T} = Union{HermOrSym{T},Factorization{T}}
```


### Functions

```@docs
lsqrt(C::Cholesky)
stein(Σ, Φ::AbstractMatrix)
stein(Σ, Φ::AbstractMatrix, Q)
schur_reduce(Π, C::AbstractMatrix)
schur_reduce(Π, C::AbstractMatrix, R)
```