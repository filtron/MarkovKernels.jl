# General 

```@meta
CurrentModule = MarkovKernels
```

MarkovKernels.jl supports covariance matrices of the following represenration:

```julia
const RSym{T,S} = Symmetric{T,S} where {T<:Real,S}
const CHerm{T,S} = Hermitian{T,S} where {T<:Complex,S}
const SelfAdjoint{T,S} = Union{RSym{T,S},CHerm{T,S}} where {T,S}

const CovarianceParameter{T} = Union{SelfAdjoint{T},Factorization{T}}
```