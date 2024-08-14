
abstract type PSDTrait end
struct IsPSDParametrization <: PSDTrait end
struct IsNotPSDParametrization <: PSDTrait end

ispsdparametrization(::Any) = IsNotPSDParametrization()

"""
    psdparamtrization(::Type{T}, P)

Wraps P in a psd paramtrization of eltype T.
If P is already a type of psd paramtrization, then just the eltype is converted.
"""
function psdparamtrization(::Type{T}, P) where {T} end

psdparametrization(P) = psdparametrization(eltype(P), P)

include("utils.jl")
include("selfadjoint.jl")
include("cholesky.jl")
include("scalar.jl")

const CovarianceParameter{T} = Union{SelfAdjoint{T},Factorization{T}}

stein(Σ, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ, A::AbstractAffineMap, Q) = stein(Σ, slope(A), Q)

schur_reduce(Π, C::AbstractAffineMap) = schur_reduce(Π, slope(C))
schur_reduce(Π, C::AbstractAffineMap, R) = schur_reduce(Π, slope(C), R)
