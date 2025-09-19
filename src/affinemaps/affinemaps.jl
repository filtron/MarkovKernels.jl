"""
    AbstractAffineMap{T<:Number}

Abstract type for representing affine maps between vector spaces over the field determined by T.
"""
abstract type AbstractAffineMap{T<:Number} end

include("affinemap.jl")
include("linearmap.jl")
include("affinecorrector.jl")
include("base_overloads.jl")
include("linearalgebra_overloads.jl")

eltype(::AbstractAffineMap{T}) where {T} = T

"""
    (a::AbstractAffineMap)(x)

Evaluates the affine map a at x.
"""
(a::AbstractAffineMap)(x) = slope(a) * x + intercept(a)

function (a::AbstractAffineMap)(y, x)
    y .= intercept(a)
    mul!(y, slope(a), x, true, true)
    return y
end

"""
    compose(a2::AbstractAffineMap, a1::AbstractAffineMap)

Computes the affine map a3 resulting from the composition a2 ∘ a1.

See also [`∘`](@ref)
"""
compose(a2::AbstractAffineMap, a1::AbstractAffineMap) =
    AffineMap(slope(a2) * slope(a1), slope(a2) * intercept(a1) + intercept(a2))

"""
    ∘(a2::AbstractAffineMap, a1::AbstractAffineMap)

Equivalent to compose(a2::AbstractAffineMap, a1::AbstractAffineMap).

See also [`compose`](@ref)
"""
∘(a2::AbstractAffineMap, a1::AbstractAffineMap) = compose(a2, a1)

AbstractAffineMap{T}(a::AbstractAffineMap{T}) where {T} = a
convert(::Type{T}, a::T) where {T<:AbstractAffineMap} = a
convert(::Type{T}, a::AbstractAffineMap) where {T<:AbstractAffineMap} = T(a)::T
