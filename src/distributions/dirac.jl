# maybe dont need Abstract Dirac ?
abstract type AbstractDirac{T} <: AbstractDistribution{T} end

==(D1::T, D2::T) where {T<:AbstractDirac} =
    all(f -> getfield(D1, f) == getfield(D2, f), 1:nfields(D1))

struct Dirac{T,U} <: AbstractDirac{T}
    μ::U
    Dirac{T}(μ) where {T} = new{T,typeof(μ)}(μ)
end

Dirac(μ::AbstractVector) = Dirac{eltype(μ)}(μ)
Dirac{T}(D::Dirac{U,V}) where {T,U,V<:AbstractVector} =
    T <: Real && U <: Real || T <: Complex && U <: Complex ?
    Dirac(convert(AbstractVector{T}, D.μ)) :
    error(
        "The constructor type $(T) and the argument type $(U) must both be real or both be complex",
    )

AbstractDistribution{T}(D::AbstractDirac) where {T} = AbstractDirac{T}(D)
AbstractDirac{T}(D::AbstractDirac{T}) where {T} = D
AbstractDirac{T}(D::Dirac) where {T} = Dirac{T}(D)

dim(D::Dirac) = length(D.μ)
mean(D::Dirac) = D.μ
cov(D::Dirac) = Diagonal(zeros(eltype(D), dim(D)))
var(D::Dirac) = zeros(real(eltype(D)), dim(D))
std(D::Dirac) = zeros(real(eltype(D)), dim(D))

rand(::AbstractRNG, D::Dirac) = mean(D)
rand(D::Dirac) = rand(GLOBAL_RNG, D)
