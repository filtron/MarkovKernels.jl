"""
    AbstractDiracKernel{T<:Number}

Abstract type for representing Dirac kernels taking values in T.
"""
abstract type AbstractDiracKernel{T} <: AbstractMarkovKernel{T} end

"""
    DiracKernel

Type for representing Dirac kernels K(y,x) = δ(y - μ(x)).
"""
struct DiracKernel{T,U} <: AbstractDiracKernel{T}
    μ::U
    DiracKernel{T}(μ) where {T} = new{T,typeof(μ)}(μ)
end

"""
    DiracKernel(F::AbstractAffineMap)

Creates a DiracKernel with conditional mean function F.
"""
DiracKernel(F::AbstractAffineMap) = DiracKernel{eltype(F)}(F)

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

const AffineDiracKernel{T} = DiracKernel{T,<:AbstractAffineMap}

"""
    DiracKernel{T}(K::AffineDiracKernel{U}) where {T,U}

Computes a Dirac kernel of eltype T from the Dirac kernel K if T and U are compatible.
That is T and U must both be Real or both be Complex.
"""
DiracKernel{T}(K::AffineDiracKernel{U}) where {T,U} =
    T <: Real && U <: Real || T <: Complex && U <: Complex ?
    DiracKernel(convert(AbstractAffineMap{T}, K.μ)) :
    error("T and U must both be complex or both be real")

AbstractMarkovKernel{T}(K::AbstractDiracKernel) where {T} = AbstractDiracKernel{T}(K)
AbstractDiracKernel{T}(K::AbstractDiracKernel{T}) where {T} = K
AbstractDiracKernel{T}(K::DiracKernel) where {T} = DiracKernel{T}(K)

"""
    mean(K::AbstractDiracKernel)

Computes the conditonal mean function of the Dirac kernel K.
That is, the output is callable.
"""
mean(K::DiracKernel) = K.μ

"""
    condition(K::AbstractDiracKernel, x)

Returns a Dirac distribution corresponding to the Dirac kernel K evaluated at x.
"""
condition(K::DiracKernel, x) = Dirac(mean(K)(x))

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

function Base.show(io::IO, N::DiracKernel{T,U}) where {T,U}
    print(io, "DiracKernel{$T,$U}(μ)")
    print(io, "\n μ = ")
    show(io, N.μ)
end
