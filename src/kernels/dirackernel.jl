# maybe dont need Abstract Dirac ?
abstract type AbstractDiracKernel{T} <: AbstractMarkovKernel{T} end

==(K1::T, K2::T) where {T<:AbstractDiracKernel} =
    all(f -> getfield(K1, f) == getfield(K2, f), 1:nfields(K1))

"""
    DiracKernel

Type for representing Dirac kernels K(y,x) = δ(y - μ(x))
"""
struct DiracKernel{T,U} <: AbstractDiracKernel{T}
    μ::U
    DiracKernel{T}(μ) where {T} = new{T,typeof(μ)}(μ)
end

DiracKernel(F::AbstractAffineMap) = DiracKernel{eltype(F)}(F)
DiracKernel(Φ::AbstractMatrix) = DiracKernel(LinearMap(Φ))
DiracKernel(Φ::AbstractMatrix, b::AbstractVector) = DiracKernel(AffineMap(Φ, b))
DiracKernel(Φ::AbstractMatrix, b::AbstractVector, c::AbstractVector) =
    DiracKernel(AffineCorrector(Φ, b, c))

const AffineDiracKernel{T} = DiracKernel{T,<:AbstractAffineMap}

DiracKernel{T}(K::DiracKernel{U,V}) where {T,U,V<:AbstractAffineMap} =
    T <: Real && U <: Real || T <: Complex && U <: Complex ?
    DiracKernel(convert(AbstractAffineMap{T}, K.μ)) :
    error("T and U must both be complex or both be real")

AbstractMarkovKernel{T}(K::AbstractDiracKernel) where {T} = AbstractDiracKernel{T}(K)
AbstractDiracKernel{T}(K::AbstractDiracKernel{T}) where {T} = K
AbstractDiracKernel{T}(K::DiracKernel) where {T} = DiracKernel{T}(K)

mean(K::DiracKernel) = K.μ
cov(K::AffineDiracKernel{T}) where {T} = x -> Diagonal(zeros(T, nout(mean(K))))

condition(K::DiracKernel, x) = Dirac(mean(K)(x))

rand(RNG::AbstractRNG, K::AbstractDiracKernel, x::AbstractVector) = mean(condition(K, x))
rand(K::AbstractDiracKernel, x::AbstractVector) = rand(GLOBAL_RNG, K, x)


function Base.show(io::IO, N::DiracKernel{T,U}) where {T,U} 
    print(io, "DiracKernel{$T,$U}(μ)")
    print(io, "\n μ = ") 
    show(io, N.μ)
end