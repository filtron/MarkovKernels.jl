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

"""
    ParticleSystem(logws::AbstractVector{<:Real}, X::AbstractArray{<:AbstractVector{T}})

Creates a ParticleSystem with logarithm of the mixture weights given by logws and location parameters in X.
If X is an AbstractVector the resulting object represents a classical Dirac mixtrue.
Whereas if X is an AbstractMatrix, the resulting object represents a Dirac mixture over trajectories,
where the row dimension represents time and the column dimension enumerates the particles.
"""
function ParticleSystem(
    logws::AbstractVector{<:Real},
    X::AbstractArray{<:AbstractVector{T}},
) where {T}
    # check dimension match
    return ParticleSystem{T,typeof(logws),typeof(X)}(logws, X)
end


Base.copy(P::ParticleSystem) =  ParticleSystem(copy(P.logws), copy(P.X))
function Base.copy!(Pdst::ParticleSystem, Psrc::ParticleSystem)
    copy!(Pdst.logws, Psrc.logws)
    copy!(Pdst.X, Pdst.X)
    return Pdst
end
Base.similar(P::ParticleSystem) =  ParticleSystem(similar(P.logws), similar(P.X))


"""
    dim(P::AbstractParticleSystem)

Returns the dimension of the particle system distribution P.
"""
dim(P::ParticleSystem) = unique(length.(particles(P)))[1]

"""
    logweights(P::AbstractParticleSystem)

Returns the logarithms of the mixture weights of the particle system P.
"""
logweights(P::ParticleSystem) = P.logws

"""
    weights(P::AbstractParticleSystem)

Returns the mixture weights of the particle system P.
"""
function weights(P::ParticleSystem)
    logc = maximum(logweights(P))
    logws = logweights(P) .- logc
    return exp.(logws) / sum(exp, logws)
end

"""
    nparticles(P::AbstractParticleSystem)

Computes the number of particles in the particle system P.
"""
nparticles(P::ParticleSystem) = length(logweights(P))

"""
    particles(P::AbstractParticleSystem)

Returns the particle locations of the particle system P.
"""
particles(P::ParticleSystem) = P.X

"""
    mean(P::AbstractParticleSystem)

Computes the mean of the particle system distribution P.
"""
mean(P::ParticleSystem) = reduce(hcat, P.X) * exp.(logweights(P))
