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

Base.iterate(F::AffineMap) = (F.A, Val(:b))
Base.iterate(F::AffineMap, ::Val{:b}) = (F.b, Val(:done))
Base.iterate(::AffineMap, ::Val{:done}) = nothing

"""
    slope(F::AbstractAffineMap)

Computes the slope of F.
"""
slope(F::AffineMap) = F.A

"""
    intercept(F::AffineMap)

Computes the intercept of F.
"""
intercept(F::AffineMap) = F.b

AffineMap{T}(F::AffineMap) where {T} =
    AffineMap(convert(AbstractMatrix{T}, F.A), convert(AbstractVector{T}, F.b))
AffineMap{T}(F::AffineMap{<:Number,<:Adjoint,<:Number}) where {T} =
    AffineMap(convert(AbstractMatrix{T}, F.A), convert(T, F.b))
AffineMap{T}(F::AffineMap{<:Number,<:Number,<:Number}) where {T} =
    AffineMap(convert(T, F.A), convert(T, F.b))

"""
    AbstractAffineMap{T}(F::AbstractAffineMap)

Creates an affine map from F with eltype T.
"""
AbstractAffineMap{T}(F::AffineMap) where {T} = AffineMap{T}(F)

function Base.show(io::IO, F::AffineMap{T,U,V}) where {T,U,V}
    print(io, summary(F))
    print(io, "\n A = ")
    show(io, (F.A))
    print(io, "\n b = ")
    show(io, F.b)
end
