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

Base.iterate(F::AffineCorrector) = (F.A, Val(:b))
Base.iterate(F::AffineCorrector, ::Val{:b}) = (F.b, Val(:c))
Base.iterate(F::AffineCorrector, ::Val{:c}) = (F.c, Val(:done))
Base.iterate(::AffineCorrector, ::Val{:done}) = nothing

slope(F::AffineCorrector) = F.A
intercept(F::AffineCorrector) = F.b - F.A * F.c

(F::AffineCorrector)(x) = F.b + slope(F) * (x - F.c)
compose(F2::AffineCorrector, F1::AffineCorrector) =
    AffineCorrector(F2.A * F1.A, F2(F1.b), F1.c)

AffineCorrector{T}(F::AffineCorrector) where {T} = AffineCorrector(
    convert(AbstractMatrix{T}, F.A),
    convert(AbstractVector{T}, F.b),
    convert(AbstractVector{T}, F.c),
)
AbstractAffineMap{T}(F::AffineCorrector) where {T} = AffineCorrector{T}(F)

function Base.show(io::IO, F::AffineCorrector{T,U,V,S}) where {T,U,V,S}
    print(io, summary(F))
    print(io, "\n A = ")
    show(io, (F.A))
    print(io, "\n b = ")
    show(io, F.b)
    print(io, "\n c = ")
    show(io, F.c)
end
