"""
    AffineMap{T,U,V}

Type for representing affine maps in the standard slope / intercept parametrisation.
"""
struct AffineMap{T,U,V} <: AbstractAffineMap{T}
    A::U
    b::V
    AffineMap{T,U,V}(A, b) where {T,U,V} = new{T,U,V}(A, b)
end

function AffineMap{T}(A::AbstractMatrix, b::AbstractVector) where {T}
    A = convert(AbstractMatrix{T}, A)
    b = convert(AbstractVector{T}, b)
    AffineMap{T,typeof(A),typeof(b)}(A, b)
end

function AffineMap{T}(A::Adjoint{<:Number,<:AbstractVector}, b::Number) where {T}
    A = convert(AbstractMatrix{T}, A)
    b = convert(T, b)
    AffineMap{T,typeof(A),typeof(b)}(A, b)
end

function AffineMap{T}(A::AbstractVector, b::AbstractVector) where {T}
    A = convert(AbstractVector{T}, A)
    b = convert(AbstractVector{T}, b)
    AffineMap{T,typeof(A),typeof(b)}(A, b)
end

function AffineMap{T}(A::Number, b::Number) where {T}
    A = convert(T, A)
    b = convert(T, b)
    AffineMap{T,typeof(A),typeof(b)}(A, b)
end

"""
    AffineMap(A, b)

Creates an AffineMap with slope A and intercept b.
"""
function AffineMap(A, b)
    T = promote_type(eltype(A), eltype(b))
    AffineMap{T}(A, b)
end

Base.iterate(a::AffineMap) = (a.A, Val(:b))
Base.iterate(a::AffineMap, ::Val{:b}) = (a.b, Val(:done))
Base.iterate(::AffineMap, ::Val{:done}) = nothing

"""
    slope(a::AbstractAffineMap)

Computes the slope of a.
"""
slope(a::AffineMap) = a.A

"""
    intercept(a::AffineMap)

Computes the intercept of a.
"""
intercept(a::AffineMap) = a.b

(a::AffineMap)(x) = slope(a) * x + intercept(a)

function (a::AffineMap)(y, x)
    y .= intercept(a)
    mul!(y, slope(a), x, true, true)
    return y
end

AffineMap{T}(a::AffineMap) where {T} =
    AffineMap(convert(AbstractMatrix{T}, a.A), convert(AbstractVector{T}, a.b))
AffineMap{T}(a::AffineMap{<:Number,<:Adjoint,<:Number}) where {T} =
    AffineMap(convert(AbstractMatrix{T}, a.A), convert(T, a.b))
AffineMap{T}(a::AffineMap{<:Number,<:Number,<:Number}) where {T} =
    AffineMap(convert(T, a.A), convert(T, a.b))

"""
    AbstractAffineMap{T}(F::AbstractAffineMap)

Creates an affine map from F with eltype T.
"""
AbstractAffineMap{T}(a::AffineMap) where {T} = AffineMap{T}(a)

function Base.show(io::IO, a::AffineMap{T,U,V}) where {T,U,V}
    print(io, summary(a))
    print(io, "\n A = ")
    show(io, (a.A))
    print(io, "\n b = ")
    show(io, a.b)
end
