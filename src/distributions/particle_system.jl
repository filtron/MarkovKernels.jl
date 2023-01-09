"""
    AbstractParticleSystem{T} <: AbstractDistribution{T}

abstract type for representing systems of particles.
"""
abstract type AbstractParticleSystem{T} <: AbstractDistribution{T} end

"""
    ParticleSystem{T,A,B} <: AbstractParticleSystem{T}

type for representing a standard particle system.
"""
struct ParticleSystem{T,A,B} <: AbstractParticleSystem{T}
    logws::A
    X::B
end

# constructor for time marginals
# X::AbstractMatrix -> trajectories
# X::AbstractVector -> time marginals
function ParticleSystem(
    logws::AbstractVector{<:Real},
    X::AbstractArray{<:AbstractVector{T}},
) where {T}
    # check dimension match
    return ParticleSystem{T,typeof(logws),typeof(X)}(logws, X)
end

dim(P::ParticleSystem) = unique(length.(particles(P)))[1]
logweights(P::ParticleSystem) = P.logws

function weights(P::ParticleSystem)
    logc = maximum(logweights(P))
    logws = logweights(P) .- logc
    return exp.(logws) / sum(exp, logws)
end

nparticles(P::ParticleSystem) = length(logweights(P))
particles(P::ParticleSystem) = P.X

mean(P::ParticleSystem) = reduce(hcat, P.X) * exp.(logweights(P))
