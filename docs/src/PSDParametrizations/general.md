# General

```@meta
CurrentModule = MarkovKernels
```

MarkovKernels.jl currently implements functionality for covariance matrices of the following representation:

```julia
const RSym{T,S} = Symmetric{T,S} where {T<:Real,S}
const CHerm{T,S} = Hermitian{T,S} where {T<:Complex,S}
const SelfAdjoint{T,S} = Union{RSym{T,S},CHerm{T,S}} where {T,S}

const CovarianceParameter{T} = Union{SelfAdjoint{T},Factorization{T}}
```

```@docs
convert_psd_eltype(::Type{T}, P) where {T}
```