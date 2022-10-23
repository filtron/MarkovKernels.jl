abstract type AbstractAffineMap{T<:Number} end

eltype(::AbstractAffineMap{T}) where {T} = T

(F::AbstractAffineMap)(x) = slope(F) * x + intercept(F)
compose(F2::AbstractAffineMap, F1::AbstractAffineMap) =
    AffineMap(slope(F2) * slope(F1), slope(F2) * intercept(F1) + intercept(F2))
*(F2::AbstractAffineMap, F1::AbstractAffineMap) = compose(F2, F1)
nout(F::AbstractAffineMap) = size(slope(F), 1)

AbstractAffineMap{T}(F::AbstractAffineMap{T}) where {T} = F
convert(::Type{T}, F::T) where {T<:AbstractAffineMap} = F
convert(::Type{T}, F::AbstractAffineMap) where {T<:AbstractAffineMap} = T(F)::T

# this falls back to === in Base if F1 and F2 are not of same eltype?
==(F1::T, F2::T) where {T<:AbstractAffineMap} =
    all(f -> getfield(F1, f) == getfield(F2, f), 1:nfields(F1))

struct AffineMap{T,U,V} <: AbstractAffineMap{T}
    A::U
    b::V
    function AffineMap(A::AbstractMatrix, b::AbstractVector)
        T = promote_type(eltype(A), eltype(b))
        A = convert(AbstractMatrix{T}, A)
        b = convert(AbstractVector{T}, b)
        new{T,typeof(A),typeof(b)}(A, b)
    end
end

Base.iterate(F::AffineMap) = (F.A, Val(:b))
Base.iterate(F::AffineMap, ::Val{:b}) = (F.b, Val(:done))
Base.iterate(F::AffineMap, ::Val{:done}) = nothing

slope(F::AffineMap) = F.A
intercept(F::AffineMap) = F.b

AffineMap{T}(F::AffineMap) where {T} =
    AffineMap(convert(AbstractMatrix{T}, F.A), convert(AbstractVector{T}, F.b))
AbstractAffineMap{T}(F::AffineMap) where {T} = AffineMap{T}(F)

struct LinearMap{T,U} <: AbstractAffineMap{T}
    A::U
    LinearMap(A::AbstractMatrix) = new{eltype(A),typeof(A)}(A)
end

Base.iterate(F::LinearMap) = (F.A, Val(:done))
Base.iterate(F::LinearMap, ::Val{:done}) = nothing

slope(F::LinearMap) = F.A
intercept(F::LinearMap) = zeros(eltype(F), size(slope(F), 1))
(F::LinearMap)(x) = slope(F) * x
compose(F2::LinearMap, F1::LinearMap) = LinearMap(slope(F2) * slope(F1))

LinearMap{T}(F::LinearMap) where {T} = LinearMap(convert(AbstractMatrix{T}, F.A))
AbstractAffineMap{T}(F::LinearMap) where {T} = LinearMap{T}(F)

struct AffineCorrector{T,U,V,S} <: AbstractAffineMap{T}
    A::U
    b::V
    c::S
    function AffineCorrector(A::AbstractMatrix, b::AbstractVector, c::AbstractVector)
        T = promote_type(eltype(A), eltype(b), eltype(c))
        A = convert(AbstractMatrix{T}, A)
        b = convert(AbstractVector{T}, b)
        c = convert(AbstractVector{T}, c)
        new{T,typeof(A),typeof(b),typeof(c)}(A, b, c)
    end
end

Base.iterate(F::AffineCorrector) = (F.A, Val(:b))
Base.iterate(F::AffineCorrector, ::Val{:b}) = (F.b, Val(:c))
Base.iterate(F::AffineCorrector, ::Val{:c}) = (F.c, Val(:done))
Base.iterate(F::AffineCorrector, ::Val{:done}) = nothing

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
