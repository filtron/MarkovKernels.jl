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
stein(Σ::HermOrSym, Φ::AbstractMatrix)
stein(Σ::HermOrSym, Φ::AbstractMatrix, Q::HermOrSym)
schur_reduce(Π::HermOrSym, C::AbstractMatrix)
schur_reduce(Π::HermOrSym, C::AbstractMatrix, R::HermOrSym)
```