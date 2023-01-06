
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

"""
    AbstractParticleSystem{T} <: AbstractDistribution{T}

Abstract type for representing systems of particles. 
"""
abstract type AbstractParticleSystem{T} <: AbstractDistribution{T} end 

"""
    ParticleSystem{T,A,B} <: AbstractParticleSystem{T}

Type for representing a standard particle system. 
"""
struct ParticleSystem{T,A,B} <: AbstractParticleSystem{T}
    weights::A 
    X::B
end

# constructor for time marginals 
# X::AbstractMatrix -> trajectories 
# X::AbstractVector -> time marginals
function ParticleSystem(
    weights::AbstractVector{<:Real},
    X::AbstractArray{<:AbstractVector{T}}
    ) where {T}
    # check dimension match
    return ParticleSystem{T,typeof(weights),typeof(X)}(weights, X)
end

weights(P::ParticleSystem) = P.weights
nparticles(P::ParticleSystem) = length(weights(P))
particles(P::ParticleSystem) = P.X 

mean(P::ParticleSystem) = reduce(hcat, P.X) * weights(P)

#update_weights!(P::ParticleSystem, )
#rand!(P::ParticleSystem, K::MarkovKernel)
