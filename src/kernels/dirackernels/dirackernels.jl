"""
    AbstractDiracKernel

Abstract type for representing Dirac kernels.
"""
abstract type AbstractDiracKernel <: AbstractMarkovKernel end

"""
    mean(K::AbstractDiracKernel)

Computes the conditonal mean function of the Dirac kernel K.
That is, the output is callable.
"""
mean(::AbstractDiracKernel)

"""
    condition(K::AbstractDiracKernel, x)

Returns a Dirac distribution corresponding to the Dirac kernel K evaluated at x.
"""
condition(K::AbstractDiracKernel, x) = Dirac(mean(K)(x))

"""
    rand(::AbstractRNG, K::AbstractDiracKernel, x::AbstractVector)

Computes a random vector conditionally on x with respect the the Dirac kernel K
using the random number generator RNG. Equivalent to mean(K)(x).
"""
rand(::AbstractRNG, K::AbstractDiracKernel, x::AbstractVector) = mean(condition(K, x))

"""
    rand(K::AbstractDiracKernel, x::AbstractVector)

Computes a random vector conditionally on x with respect the the Dirac kernel K
using the random number generator Random.default_rng(). Equivalent to mean(K)(x).
"""
rand(K::AbstractDiracKernel, x::AbstractVector) = rand(Random.default_rng(), K, x)

function Base.show(io::IO, D::AbstractDiracKernel)
    println(io, summary(D))
    println(io, "μ = ")
    show(io, mean(D))
end

include("dirackernel.jl")
include("identitykernel.jl")
