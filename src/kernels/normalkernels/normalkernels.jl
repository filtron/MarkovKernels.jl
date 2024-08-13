"""
    AbstractNormalKernel

Abstract type for representing Normal kernels.
"""
abstract type AbstractNormalKernel <: AbstractMarkovKernel end

"""
    mean(K::AbstractNormalKernel)

Computes the conditonal mean function of the Normal kernel K.
That is, the output is callable.
"""
function mean(::AbstractNormalKernel) end

"""
    covp(K::AbstractNormalKernel)

Returns the internal representation of the conditonal covariance of the Normal kernel K.
For computing the actual conditional covariance, use cov.
"""
function covp(::AbstractNormalKernel) end

"""
    cov(K::AbstractNormalKernel)

Computes the conditonal covariance function of the Normal kernel K.
That is, the output is callable.
"""
function cov(::AbstractNormalKernel) end

"""
    condition(K::AbstractNormalKernel, x)

Returns a Normal distribution corresponding to K evaluated at x.
"""
condition(K::AbstractNormalKernel, x) = Normal(mean(K)(x), cov(K)(x))

"""
    rand([rng::AbstractRNG], K::AbstractNormalKernel, x::AbstractVector)

Computes a random vector conditionally on x with respect the the Normal kernel K
using the random number generator rng.
"""
rand(rng::AbstractRNG, K::AbstractNormalKernel, x::AbstractVector) =
    rand(rng, condition(K, x))
rand(K::AbstractNormalKernel, x::AbstractVector) = rand(Random.default_rng(), K, x)

function Base.show(io::IO, K::AbstractNormalKernel)
    println(io, summary(K))
    println(io, "μ = ")
    show(io, mean(K))
    println(io, "\nΣ = ")
    show(io, covp(K))
end

include("normalkernel.jl")
include("homoskedasticnormalkernel.jl")

normalkernel(μ, Σ) = NormalKernel(μ, Σ)
normalkernel(μ, Σ::Union{AbstractMatrix,Factorization,Number}) =
    HomoskedasticNormalKernel(μ, Σ)
