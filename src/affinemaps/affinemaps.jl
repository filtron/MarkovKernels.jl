"""
    AbstractAffineMap{T<:Number}

Abstract type for representing affine maps between vector spaces over the field determined by T.
"""
abstract type AbstractAffineMap{T<:Number} end

include("affinemap.jl")
include("linearmap.jl")
include("affinecorrector.jl")

eltype(::AbstractAffineMap{T}) where {T} = T

"""
    (F::AbstractAffineMap)(x)

Evaluates the affine map F at x.
"""
(F::AbstractAffineMap)(x) = slope(F) * x + intercept(F)

"""
    compose(F2::AbstractAffineMap, F1::AbstractAffineMap)

Computes the affine map F3 resulting from the composition F2 ∘ F1.
"""
compose(F2::AbstractAffineMap, F1::AbstractAffineMap) =
    AffineMap(slope(F2) * slope(F1), slope(F2) * intercept(F1) + intercept(F2))

"""
    ∘(F2::AbstractAffineMap, F1::AbstractAffineMap)

Equivalent to compose(F2::AbstractAffineMap, F1::AbstractAffineMap).
"""
∘(F2::AbstractAffineMap, F1::AbstractAffineMap) = compose(F2, F1)

AbstractAffineMap{T}(F::AbstractAffineMap{T}) where {T} = F
convert(::Type{T}, F::T) where {T<:AbstractAffineMap} = F
convert(::Type{T}, F::AbstractAffineMap) where {T<:AbstractAffineMap} = T(F)::T

==(F1::T, F2::T) where {T<:AbstractAffineMap} =
    all(f -> getfield(F1, f) == getfield(F2, f), 1:nfields(F1))
