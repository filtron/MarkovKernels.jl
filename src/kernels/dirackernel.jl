"""
    AbstractDiracKernel

Abstract type for representing Dirac kernels.
"""
abstract type AbstractDiracKernel <: AbstractMarkovKernel end

"""
    DiracKernel

Type for representing Dirac kernels K(y,x) = δ(y - μ(x)).
"""
struct DiracKernel{L} <: AbstractDiracKernel
    μ::L
end

const AffineDiracKernel{T} = DiracKernel{<:AbstractAffineMap{T}} where {T}

"""
    IdentityKernel

Struct for representing kernels that act like identity under marginalization.
"""
struct IdentityKernel <: AbstractDiracKernel end

"""
    mean(K::AbstractDiracKernel)

Computes the conditonal mean function of the Dirac kernel K.
That is, the output is callable.
"""
mean(K::DiracKernel) = K.μ
mean(::IdentityKernel) = identity

condition(K::AbstractDiracKernel, x) = Dirac(mean(K)(x))

"""
    rand([rng::AbstractRNG], K::AbstractDiracKernel, x::AbstractVector)

Computes a random vector conditionally on x with respect the the Dirac kernel K
using the random number generator RNG. Equivalent to mean(K)(x).
"""
rand(::AbstractRNG, K::AbstractDiracKernel, x::AbstractNumOrVec) = mean(condition(K, x))
rand(K::AbstractDiracKernel, x::AbstractNumOrVec) = rand(Random.default_rng(), K, x)

function Base.show(io::IO, D::DiracKernel)
    println(io, summary(D))
    println(io, "μ = ")
    show(io, mean(D))
end
