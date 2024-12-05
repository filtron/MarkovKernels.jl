"""
    AbstractDirac{ST}

Abstract type for representing Dirac distributions taking values in ST.
"""
abstract type AbstractDirac{ST} <: AbstractDistribution{ST} end

"""
    Dirac

Type for representing Dirac distributions with sample_eltype ST.
"""
struct Dirac{ST,U} <: AbstractDirac{ST}
    μ::U
end

"""
    Dirac(μ)

Creates a Dirac distribution with mean μ.
"""
Dirac(μ) = Dirac{typeof(μ),typeof(μ)}(μ)

"""
    mean(D::AbstractDirac)

Computes the mean vector of the Dirac distribution D.
"""
mean(D::Dirac) = D.μ

"""
    dim(D::AbstractDirac)

Returns the dimension of the Dirac distribution D.
"""
dim(D::Dirac) = length(mean(D))

rand(::AbstractRNG, D::AbstractDirac) = mean(D)

function Base.show(io::IO, D::Dirac)
    println(io, summary(D))
    print(io, "μ = ")
    show(io, mean(D))
end
