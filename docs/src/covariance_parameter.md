# Covariance Parameter

```@meta
CurrentModule = MarkovKernels
```

In MarkovKernels.jl a covariance matrix is assumed to be of the following type union.

```julia
const CovarianceParameter{T} = Union{HermOrSym{T},UniformScaling{T},Factorization{T}}
```


### Functions

```@docs
lsqrt(J::UniformScaling)
lsqrt(A::AbstractMatrix)
stein(Σ, Φ::AbstractMatrix)
stein(Σ, Φ::AbstractMatrix, Q)
schur_reduce(Π, C::AbstractMatrix)
schur_reduce(Π, C::AbstractMatrix, R)
```