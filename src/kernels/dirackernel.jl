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

"""
    DiracKernel(Φ::AbstractMatrix, Σ)

Creates a DiracKernel with a linear conditional mean function given by

    x ↦ Φ * x.
"""
DiracKernel(Φ::AbstractMatrix) = DiracKernel(LinearMap(Φ))

"""
    DiracKernel(Φ::AbstractMatrix, b::AbstractVector, Σ)

Creates a DiracKernel with an affine conditional mean function given by

    x ↦ b + Φ * x.
"""
DiracKernel(Φ::AbstractMatrix, b::AbstractVector) = DiracKernel(AffineMap(Φ, b))

"""
    DiracKernel(Φ::AbstractMatrix, b::AbstractVector, c::AbstractVector, Σ)

Creates a DiracKernel with an affine corrector conditional mean function given by

    x ↦ b + Φ * (x - c).
"""
DiracKernel(Φ::AbstractMatrix, b::AbstractVector, c::AbstractVector) =
    DiracKernel(AffineCorrector(Φ, b, c))

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
using the random number generator Random.GLOBAL_RNG. Equivalent to mean(K)(x).
"""
rand(K::AbstractDiracKernel, x::AbstractVector) = rand(GLOBAL_RNG, K, x)

function Base.show(io::IO, D::DiracKernel)
    println(io, summary(D))
    println(io, "μ = ")
    show(io, D.μ)
end
