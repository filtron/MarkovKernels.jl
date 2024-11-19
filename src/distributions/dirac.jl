"""
    AbstractDirac{T<:Number}

Abstract type for representing Dirac random vectors taking values in T.
"""
abstract type AbstractDirac{T} <: AbstractDistribution{T} end

"""
    Dirac{T<:Number}

Type for representing Dirac distributions with eltype_sample T.
"""
struct Dirac{T,U} <: AbstractDirac{T}
    μ::U
end

"""
    Dirac(μ)

Creates a Dirac distribution with mean μ.
"""
Dirac(μ) = Dirac{eltype(μ),typeof(μ)}(μ)

"""
    mean(D::AbstractDirac)

Computes the mean vector of the Dirac distribution D.
"""
mean(D::Dirac) = D.μ

sample_type(D::AbstractDirac) = typeof(mean(D))

"""
    Dirac{T}(D::Dirac)

Computes a Dirac distribution of sample_eltype T from the Dirac distribution D.
"""
Dirac{T}(D::Dirac{U,<:Number}) where {T,U} = Dirac(convert(T, mean(D)))
Dirac{T}(D::Dirac{U,<:AbstractVector}) where {T,U} =
    Dirac(convert(AbstractVector{T}, mean(D)))

AbstractDistribution{T}(D::AbstractDirac) where {T} = AbstractDirac{T}(D)
AbstractDirac{T}(D::AbstractDirac{T}) where {T} = D
AbstractDirac{T}(D::Dirac) where {T} = Dirac{T}(D)

"""
    dim(D::AbstractDirac)

Returns the dimension of the Dirac distribution D.
"""
dim(D::Dirac) = length(mean(D))

"""
    rand(RNG::AbstractRNG, D::AbstractDirac)

Computes a random vector distributed according to the Dirac distribution D
using the random number generator RNG. Equivalent to mean(D).
"""
rand(::AbstractRNG, D::AbstractDirac) = mean(D)

function Base.show(io::IO, D::Dirac)
    println(io, summary(D))
    print(io, "μ = ")
    show(io, mean(D))
end
