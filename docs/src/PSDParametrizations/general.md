

```@meta
CurrentModule = MarkovKernels
```

MarkovKernels.jl aims to support a wide variety of parametrizations of positive semi-definite matrices.
How this is accomplished is explaiend in the following.



## Types

Currently, the following types are valid PSD parametrizations (in the sense, the PSDParametriszations methods below have been implemented):

```julia
Real
SelfAdjoint
Cholesky
```

Here, the definition of ```SelfAdjoint``` is:

```julia
const RealSymmetric{T,S} = Symmetric{T,S} where {T<:Real,S}
const ComplexHermitian{T,S} = Hermitian{T,S} where {T<:Complex,S}
const RealDiagonal{T,S} = Diagonal{T,S} where {T<:Real,S}
const SelfAdjoint{T,S} =
    Union{RealSymmetric{T,S},ComplexHermitian{T,S},RealDiagonal{T,S}} where {T,S}
```
## PSD Trait

In order to determine, whether a type is a ```PSDParametrization``` the following types are defined:

```julia
abstract type PSDTrait end
struct IsPSD <: PSDTrait end
struct IsNotPSD <: PSDTrait end
```

A custom typer can opt into being a ```PSDparametrization``` by implementing

```julia
psdcheck(::MyPSDType) = IsPSD()
```

which is a promise that the methods of the next section have been implemented.


## PSDParametrizations methods


```@docs
psdcheck(::Any)
convert_psd_eltype(::Type{T}, P) where {T}
rsqrt(Σ)
lsqrt(Σ)
stein(Σ, Φ, Q)
schur_reduce(Π, C, R)
```

## SelfAdjoint methods

Additionally, the following methods are defined for ```SelfAdjoint```:

```@docs
selfadjoint!(x::Number)
selfadjoint(A)
```