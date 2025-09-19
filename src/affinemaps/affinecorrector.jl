"""
    AffineCorrector{T,U,V,S}

Type for representing affine correctors, i.e.,

    x ↦ b + A * (x -c).
"""
struct AffineCorrector{T,U,V,S} <: AbstractAffineMap{T}
    A::U
    b::V
    c::S
    AffineCorrector{T,U,V,S}(A, b, c) where {T,U,V,S} = new{T,U,V,S}(A, b, c)
end

function AffineCorrector{T}(
    A::AbstractMatrix,
    b::AbstractVector,
    c::AbstractVector,
) where {T}
    A = convert(AbstractMatrix{T}, A)
    b = convert(AbstractVector{T}, b)
    c = convert(AbstractVector{T}, c)
    AffineCorrector{T,typeof(A),typeof(b),typeof(c)}(A, b, c)
end

function AffineCorrector{T}(A::Adjoint, b::Number, c::AbstractVector) where {T}
    A = convert(AbstractMatrix{T}, A)
    b = convert(T, b)
    c = convert(AbstractVector{T}, c)
    AffineCorrector{T,typeof(A),typeof(b),typeof(c)}(A, b, c)
end

function AffineCorrector{T}(A::AbstractVector, b::AbstractVector, c::Number) where {T}
    A = convert(AbstractVector{T}, A)
    b = convert(AbstractVector{T}, b)
    c = convert(T, c)
    AffineCorrector{T,typeof(A),typeof(b),typeof(c)}(A, b, c)
end

function AffineCorrector{T}(A::Number, b::Number, c::Number) where {T}
    A = convert(T, A)
    b = convert(T, b)
    c = convert(T, c)
    AffineCorrector{T,typeof(A),typeof(b),typeof(c)}(A, b, c)
end

"""
    AffineCorrector(A, b, c)

Creates an Affine Corrector with slope A and intercept b - A * c.
"""
function AffineCorrector(A, b, c)
    T = promote_type(eltype(A), eltype(b), eltype(c))
    AffineCorrector{T}(A, b, c)
end

Base.iterate(a::AffineCorrector) = (a.A, Val(:b))
Base.iterate(a::AffineCorrector, ::Val{:b}) = (a.b, Val(:c))
Base.iterate(a::AffineCorrector, ::Val{:c}) = (a.c, Val(:done))
Base.iterate(::AffineCorrector, ::Val{:done}) = nothing

slope(a::AffineCorrector) = a.A
intercept(a::AffineCorrector) = a.b - a.A * a.c

(a::AffineCorrector)(x) = a.b + slope(a) * (x - a.c)

function (a::AffineCorrector)(y, x)
    y .= a.b
    mul!(y, a.A, a.c, -one(eltype(y)), true)
    mul!(y, a.A, x, true, true)
    return y
end

compose(a2::AffineCorrector, a1::AffineCorrector) =
    AffineCorrector(a2.A * a1.A, a2(a1.b), a1.c)

AffineCorrector{T}(a::AffineCorrector) where {T} = AffineCorrector(
    convert(AbstractMatrix{T}, a.A),
    convert(AbstractVector{T}, a.b),
    convert(AbstractVector{T}, a.c),
)
AbstractAffineMap{T}(a::AffineCorrector) where {T} = AffineCorrector{T}(a)

function Base.show(io::IO, a::AffineCorrector{T,U,V,S}) where {T,U,V,S}
    print(io, summary(a))
    print(io, "\n A = ")
    show(io, (a.A))
    print(io, "\n b = ")
    show(io, a.b)
    print(io, "\n c = ")
    show(io, a.c)
end
