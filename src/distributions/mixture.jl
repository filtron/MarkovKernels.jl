
"""
    AbstractMixture <: AbstractDistribution

Abstract type for representing mixture distributions
"""

abstract type AbstractMixture{T,U<:AbstractDistribution} <: AbstractDistribution{T} end

struct Mixture{T,A,B,C} <: AbstractMixture{T,A}
    weights::B
    dists::C
end

function Mixture(
    weights::AbstractVector{<:Real},
    dists::AbstractVector{<:AbstractDistribution{T}},
) where {T}
    return Mixture{T,eltype(dists),typeof(weights),typeof(dists)}(weights, dists)
end

weights(M::Mixture) = M.weights
ncomponents(M::Mixture) = length(weights(M))
components(M::Mixture) = M.dists

mean(M::Mixture) = reduce(hcat, mean.(components(M))) * weights(M)

cov(M::Mixture{T,<:Dirac}) where {T} = _covmean(M)

function cov(M::Mixture)
    W = diagm(weights(M))
    ΔX = reduce(hcat, mean.(components(M))) .- mean(M)
    covmean = ΔX * W * ΔX'
    meancov = sum(cov.(components(M)) .* weights(M))
    return covmean + meancov
end

function _covmean(M::Mixture)
    W = diagm(weights(M))
    ΔX = reduce(hcat, mean.(components(M))) .- mean(M)
    return ΔX * W * ΔX'
end