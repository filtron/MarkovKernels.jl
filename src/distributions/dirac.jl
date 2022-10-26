"""
    AbstractDirac{T<:Number}

Abstract type for representing Dirac random vectors taking values in T.
"""
abstract type AbstractDirac{T} <: AbstractDistribution{T} end

==(D1::T, D2::T) where {T<:AbstractDirac} =
    all(f -> getfield(D1, f) == getfield(D2, f), 1:nfields(D1))

"""
    Dirac{T<:Number}

Type for representing Dirac random vectors taking values in T.
"""
struct Dirac{T,U} <: AbstractDirac{T}
    μ::U
    Dirac{T}(μ) where {T} = new{T,typeof(μ)}(μ)
end

"""
    Dirac(μ::AbstractVector)

Creates a Dirac distribution with mean vector μ.
"""
Dirac(μ::AbstractVector) = Dirac{eltype(μ)}(μ)

"""
    Dirac{T}(D::Dirac{U,V})

Computes a Dirac distribution of eltype T from the Dirac distribution D if T and U are compatible.
That is T and U must both be Real or both be Complex.
"""
Dirac{T}(D::Dirac{U,V}) where {T,U,V<:AbstractVector} =
    T <: Real && U <: Real || T <: Complex && U <: Complex ?
    Dirac(convert(AbstractVector{T}, D.μ)) :
    error(
        "The constructor type $(T) and the argument type $(U) must both be real or both be complex",
    )

AbstractDistribution{T}(D::AbstractDirac) where {T} = AbstractDirac{T}(D)
AbstractDirac{T}(D::AbstractDirac{T}) where {T} = D
AbstractDirac{T}(D::Dirac) where {T} = Dirac{T}(D)

"""
    dim(D::AbstractDirac)

Returns the dimension of the Dirac distribution D.
"""
dim(D::Dirac) = length(D.μ)

"""
    mean(D::AbstractDirac)

Computes the mean vector of the Dirac distribution D.
"""
mean(D::Dirac) = D.μ

"""
    cov(D::AbstractDirac)

Computes the covariance matrix of the Dirac distribution D.
"""
cov(D::AbstractDirac) = Diagonal(zeros(eltype(D), dim(D)))

"""
    var(D::AbstractDirac)
Computes the vector of marginal variances of the Dirac distribution D.
"""
var(D::AbstractDirac) = zeros(real(eltype(D)), dim(D))

"""
    std(N::AbstractDirac)
Computes the vector of marginal standard deviations of the Dirac distribution D.
"""
std(D::AbstractDirac) = zeros(real(eltype(D)), dim(D))

"""
    rand(RNG::AbstractRNG, D::AbstractDirac)

Computes a random vector distributed according to the Dirac distribution D
using the random number generator RNG. Equivalent to mean(D).
"""
rand(::AbstractRNG, D::AbstractDirac) = mean(D)

"""
    rand(D::AbstractDirac)

Computes a random vector distributed according to the Dirac distribution D
using the random number generator Random.GLOBAL_RNG. Equivalent to mean(D).
"""
rand(D::AbstractDirac) = rand(GLOBAL_RNG, D)

function Base.show(io::IO, N::Dirac{T,U}) where {T,U}
    print(io, "Dirac{$T,$U}(μ)")
    print(io, "\n μ = ")
    show(io, N.μ)
end
