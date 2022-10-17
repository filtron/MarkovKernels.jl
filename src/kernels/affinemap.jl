# types for representing affine conditional means
abstract type AbstractAffineMap{T<:Number} end

eltype(::AbstractAffineMap{T}) where {T} = T

(M::AbstractAffineMap)(x) = slope(M) * x + intercept(M)
compose(M2::AbstractAffineMap, M1::AbstractAffineMap) =
    AffineMap(slope(M2) * slope(M1), slope(M2) * intercept(M1) + intercept(M2))
*(M2::AbstractAffineMap, M1::AbstractAffineMap) = compose(M2, M1)
nout(M::AbstractAffineMap) = size(slope(M), 1)

convert(::Type{T}, F::T) where {T<:AbstractAffineMap} = F
convert(::Type{T}, F::AbstractAffineMap) where {T<:AbstractAffineMap} = T(F)::T

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
slope(F::AffineMap) = F.A
intercept(F::AffineMap) = F.b
==(F1::AffineMap, F2::AffineMap) = F1.A == F2.A && F1.b == F2.b

AffineMap{T}(F::AffineMap) where {T} =
    AffineMap(convert(AbstractMatrix{T}, F.A), convert(AbstractVector{T}, F.b))
AbstractAffineMap{T}(F::AffineMap{T}) where {T} = F
AbstractAffineMap{T}(F::AffineMap) where {T} = AffineMap{T}(F)

struct LinearMap{T,U} <: AbstractAffineMap{T}
    A::U
    LinearMap(A::AbstractMatrix) = new{eltype(A),typeof(A)}(A)
end
slope(F::LinearMap) = F.A
intercept(F::LinearMap) = zeros(eltype(F), size(slope(F), 1))
==(F1::LinearMap, F2::LinearMap) = F1.A == F2.A
compose(F2::LinearMap, F1::LinearMap) = LinearMap(slope(F2) * slope(F1))

LinearMap{T}(F::LinearMap) where {T} = LinearMap(convert(AbstractMatrix{T}, F.A))
AbstractAffineMap{T}(F::LinearMap{T}) where {T} = F
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
slope(F::AffineCorrector) = F.A
intercept(F::AffineCorrector) = F.b - F.A * F.c
==(F1::AffineCorrector, F2::AffineCorrector) = F1.A == F2.A && F1.b == F2.b && F1.c == F2.c
compose(F2::AffineCorrector, F1::AffineCorrector) =
    AffineCorrector(F2.A * F1.A, F2(F1.b), F1.c)

AffineCorrector{T}(F::AffineCorrector) where {T} = AffineCorrector(
    convert(AbstractMatrix{T}, F.A),
    convert(AbstractVector{T}, F.b),
    convert(AbstractVector{T}, F.c),
)
AbstractAffineMap{T}(F::AffineCorrector{T}) where {T} = F
AbstractAffineMap{T}(F::AffineCorrector) where {T} = AffineCorrector{T}(F)
