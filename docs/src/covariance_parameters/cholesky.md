# Cholesky 

```@meta
CurrentModule = MarkovKernels
```


## Methods 

```@docs
rsqrt(C::Cholesky)
lsqrt(C::Cholesky)

stein(Σ::Cholesky, Φ::AbstractMatrix)
stein(Σ::Cholesky, Φ::Adjoint{<:Number,<:AbstractVector})
stein(Σ::Cholesky, Φ::AbstractMatrix, Q::Cholesky)
stein(Σ::Cholesky, Φ::Adjoint{<:Number,<:AbstractVector}, Q::Number)

schur_reduce(Π::Cholesky, C::AbstractMatrix)
schur_reduce(Π::Cholesky, C::Adjoint{<:Number,<:AbstractVector})
schur_reduce(Π::Cholesky, C::AbstractMatrix, R::Cholesky)
schur_reduce(Π::Cholesky, C::Adjoint{<:Number,<:AbstractVector}, R::Number)
```
