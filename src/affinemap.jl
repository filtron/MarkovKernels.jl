"""
    AbstractAffineMap{T<:Number}

Abstract type for representing affine maps between vector spaces over the field determined by T.
"""
abstract type AbstractAffineMap{T<:Number} end

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
    *(F2::AbstractAffineMap, F1::AbstractAffineMap)

Equivalent to compose(F2::AbstractAffineMap, F1::AbstractAffineMap).
"""
*(F2::AbstractAffineMap, F1::AbstractAffineMap) = compose(F2, F1)

AbstractAffineMap{T}(F::AbstractAffineMap{T}) where {T} = F
convert(::Type{T}, F::T) where {T<:AbstractAffineMap} = F
convert(::Type{T}, F::AbstractAffineMap) where {T<:AbstractAffineMap} = T(F)::T

==(F1::T, F2::T) where {T<:AbstractAffineMap} =
    all(f -> getfield(F1, f) == getfield(F2, f), 1:nfields(F1))

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

"""
    AffineMap(A::AbstractMatrix, b::AbstractVector)

Creates an AffineMap with slope A and intercept b.
"""
function AffineMap(A::AbstractMatrix, b::AbstractVector)
    T = promote_type(eltype(A), eltype(b))
    AffineMap{T}(A, b)
end

Base.iterate(F::AffineMap) = (F.A, Val(:b))
Base.iterate(F::AffineMap, ::Val{:b}) = (F.b, Val(:done))
Base.iterate(F::AffineMap, ::Val{:done}) = nothing

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

Base.copy(F::AffineMap) = AffineMap(copy(F.A), copy(F.b))
function Base.copy!(Fdst::AffineMap, Fsrc::AffineMap) 
    copy!(Fdst.A, Fsrc.)
end
Base.similar(F::AffineMap) = AffineMap(similar(F.A), similar(F.b))

"""
    AbstractAffineMap{T}(F::AbstractAffineMap)

Creates an affine map from F with eltype T.
"""
AbstractAffineMap{T}(F::AffineMap) where {T} = AffineMap{T}(F)

"""
    LinearMap{T,U}

Type for representing affine maps with zero intercept.
"""
struct LinearMap{T,U} <: AbstractAffineMap{T}
    A::U
end

"""
    LinearMap(A::AbstractMatrix)

Creates a LinearMap with slope A.
"""
LinearMap(A::AbstractMatrix) = LinearMap{eltype(A),typeof(A)}(A)

Base.iterate(F::LinearMap) = (F.A, Val(:done))
Base.iterate(F::LinearMap, ::Val{:done}) = nothing

Base.copy(F::LinearMap) = LinearMap(copy(F.A))
Base.similar(F::LinearMap) = LinearMap(similar(F.A))

slope(F::LinearMap) = F.A
intercept(F::LinearMap) = zero(diag(slope(F)))

(F::LinearMap)(x) = slope(F) * x
compose(F2::LinearMap, F1::LinearMap) = LinearMap(slope(F2) * slope(F1))

LinearMap{T}(F::LinearMap) where {T} = LinearMap(convert(AbstractMatrix{T}, F.A))
AbstractAffineMap{T}(F::LinearMap) where {T} = LinearMap{T}(F)

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

"""
    AffineCorrector(A::AbstractMatrix, b::AbstractVector, c::AbstractVector)

Creates an Affine Corrector with slope A and intercept b - A * c.
"""
function AffineCorrector(A::AbstractMatrix, b::AbstractVector, c::AbstractVector)
    T = promote_type(eltype(A), eltype(b), eltype(c))
    AffineCorrector{T}(A, b, c)
end

Base.iterate(F::AffineCorrector) = (F.A, Val(:b))
Base.iterate(F::AffineCorrector, ::Val{:b}) = (F.b, Val(:c))
Base.iterate(F::AffineCorrector, ::Val{:c}) = (F.c, Val(:done))
Base.iterate(F::AffineCorrector, ::Val{:done}) = nothing

Base.copy(F::AffineCorrector) = AffineCorrector(copy(F.A), copy(F.b), copy(F.c))
Base.similar(F::AffineCorrector) = AffineCorrector(similar(F.A), similar(F.b), similar(F.c))

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

function Base.show(io::IO, F::AffineMap{T,U,V}) where {T,U,V}
    print(io, "AffineMap{$T,$U,$V}(A, b)")
    print(io, "\n A = ")
    show(io, (F.A))
    print(io, "\n b = ")
    show(io, F.b)
end

function Base.show(io::IO, F::LinearMap{T,U}) where {T,U}
    print(io, "LinearMap{$T,$U}(A)")
    print(io, "\n A = ")
    show(io, (F.A))
end

function Base.show(io::IO, F::AffineCorrector{T,U,V,S}) where {T,U,V,S}
    print(io, "AffineCorrector{$T,$U,$V,$S}(A, b, c)")
    print(io, "\n A = ")
    show(io, (F.A))
    print(io, "\n b = ")
    show(io, F.b)
    print(io, "\n c = ")
    show(io, F.c)
end
