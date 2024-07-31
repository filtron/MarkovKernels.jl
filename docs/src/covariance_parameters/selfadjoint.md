# SelfAdjoint 

```@meta
CurrentModule = MarkovKernels
```

Self-adjoint matrices are subtypes of the following union: 

```julia
const RSym{T,S} = Symmetric{T,S} where {T<:Real,S}
const CHerm{T,S} = Hermitian{T,S} where {T<:Complex,S}
const SelfAdjoint{T,S} = Union{RSym{T,S},CHerm{T,S}} where {T,S}
```

## Methods 

```@docs
selfadjoint(x::Number)
selfadjoint(A::AbstractMatrix{<:Real})
selfadjoint(A::AbstractMatrix{<:Complex})

rsqrt(A::SelfAdjoint)
lsqrt(A::SelfAdjoint)

stein(Σ::SelfAdjoint, Φ::AbstractMatrix)
stein(Σ::SelfAdjoint, Φ::AbstractMatrix, Q::SelfAdjoint)
stein(Σ::SelfAdjoint, Φ::AbstractMatrix, Q::Real)

schur_reduce(Π::SelfAdjoint, C::AbstractMatrix)
schur_reduce(Π::SelfAdjoint, C::AbstractMatrix, R::SelfAdjoint)
schur_reduce(Π::SelfAdjoint, C::AbstractMatrix, R::Real)
```