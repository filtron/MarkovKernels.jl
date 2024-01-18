"""
    AbstractDirac{T<:Number}

Abstract type for representing Dirac random vectors taking values in T.
"""
abstract type AbstractDirac{T} <: AbstractDistribution{T} end

"""
    Dirac{T<:Number}

Type for representing Dirac random vectors taking values in T.
"""
struct Dirac{T,U} <: AbstractDirac{T}
    μ::U
end

"""
    Dirac(μ::AbstractVector{<:Number})

Creates a vector-valued Dirac distribution with mean vector μ.
"""
Dirac(μ::AbstractVector{T}) where {T<:Number} = Dirac{T,typeof(μ)}(μ)

"""
    Dirac(μ::AbstractVector{<:AbstractVector{T}}) where {T<:Number}

Creates a trajectory-valued Dirac distribution with mean trajectory μ
"""
Dirac(μ::AbstractVector{<:AbstractVector{T}}) where {T<:Number} = Dirac{T,typeof(μ)}(μ)

"""
    Dirac{T}(D::Dirac{U,V})

Computes a Dirac distribution of eltype T from the Dirac distribution D if T and U are compatible.
That is T and U must both be Real or both be Complex.
"""
function Dirac{T}(D::Dirac{U,V}) where {T,U,V<:AbstractVector}
    if T <: Real && U <: Real || T <: Complex && U <: Complex
        Dirac(convert(AbstractVector{T}, D.μ))
    else
        error(
            "The constructor type $(T) and the argument type $(U) must both be real or both be complex",
        )
    end
end

function Dirac{T}(D::Dirac{U,V}) where {T,U,V<:AbstractVector{<:AbstractVector}}
    if T <: Real && U <: Real || T <: Complex && U <: Complex
        Dirac(convert.(AbstractVector{T}, D.μ))
    else
        error(
            "The constructor type $(T) and the argument type $(U) must both be real or both be complex",
        )
    end
end

AbstractDistribution{T}(D::AbstractDirac) where {T} = AbstractDirac{T}(D)
AbstractDirac{T}(D::AbstractDirac{T}) where {T} = D
AbstractDirac{T}(D::Dirac) where {T} = Dirac{T}(D)

typeof_sample(D::Dirac) = typeof(D.μ)

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

function Base.show(io::IO, D::Dirac)
    println(io, summary(D))
    print(io, "μ = ")
    show(io, D.μ)
end
