
abstract type PSDTrait end
struct IsPSD <: PSDTrait end
struct IsNotPSD <: PSDTrait end

psdcheck(::Any) = IsNotPSD()

"""
    convert_psd_eltype(::Type{T}, P)

Wraps P in a psd paramtrization of eltype T.
If P is already a type of psd paramtrization, then just the eltype is converted.
"""
function convert_psd_eltype(::Type{T}, P) where {T} end

convert_psd_eltype(P) = convert_psd_eltype(eltype(P), P)

include("utils.jl")
include("interface.jl")
include("selfadjoint.jl")
include("cholesky.jl")
include("scalar.jl")
include("diagonal.jl")

stein(Σ, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ, A::AbstractAffineMap, Q) = stein(Σ, slope(A), Q)

schur_reduce(Π, C::AbstractAffineMap) = schur_reduce(Π, slope(C))
schur_reduce(Π, C::AbstractAffineMap, R) = schur_reduce(Π, slope(C), R)
