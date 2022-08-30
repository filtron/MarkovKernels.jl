abstract type AbstractDirac{T<:Number} <: AbstractDistribution end

eltype(::AbstractDirac{T}) where {T} = T

struct Dirac{T,U} <: AbstractDirac{T}
    μ::U
    Dirac(μ::AbstractVector) = new{eltype(μ),typeof(μ)}(μ)
end

similar(D::Dirac) = Dirac(similar(D.μ))
==(D1::Dirac, D2::Dirac) = D1.μ == D2.μ

dim(D::Dirac) = length(D.μ)
mean(D::Dirac) = D.μ
cov(D::Dirac) = zeros(eltype(D), dim(D), dim(D))
var(D::Dirac) = zeros(real(eltype(D)), dim(D))
std(D::Dirac) = zeros(real(eltype(D)), dim(D))

rand(RNG::AbstractRNG, D::Dirac) = mean(D)
rand(D::Dirac) = rand(GLOBAL_RNG, D)
