"""
    AbstractStochasticMatrix

Abstract type for representing stochastic matrices.
"""
abstract type AbstractStochasticMatrix <: AbstractMarkovKernel end

"""
    probability_vector(::AbstractCategorical)


Computes the matrix of transition probabilities.
"""
function probability_matrix(::AbstractStochasticMatrix) end

"""
    StochasticMatrix

Type for representing stochastic matrices.
"""
struct StochasticMatrix{A} <: AbstractStochasticMatrix
    P::A
end

"""
StochasticMatrix(P::AbstractMatrix)

Constructs a stochastic matrix from the matrix of transition probabilities P.
"""
function StochasticMatrix(P::AbstractMatrix, normalize = true)
    if normalize
        Π = copy(P)
        foreach(Base.Fix2(normalize!, 1), eachcol(P))
    else
        Π = P
    end
    return StochasticMatrix{typeof(Π)}(Π)
end

probability_matrix(K::StochasticMatrix) = K.P

condition(K::AbstractStochasticMatrix, x) = ProbabilityVector(K.P[:, x])

"""
    rand([rng::AbstractRNG], K::AbstractStochasticMatrix, x)

Samples a random vector conditionally on x with respect the the stochastic matrix K
using the random number generator rng.
"""
rand(rng::AbstractRNG, K::AbstractStochasticMatrix, x::Int) = rand(rng, condition(K, x))
rand(K::AbstractStochasticMatrix, x::Int) = rand(Random.default_rng(), K, x)

eltype(k::AbstractStochasticMatrix) = eltype(probability_matrix(k))

size(k::AbstractStochasticMatrix) = size(probability_matrix(k))
size(k::AbstractStochasticMatrix, i) = size(k)[i]

function Base.copy!(kdst::AbstractStochasticMatrix, ksrc::AbstractStochasticMatrix)
    copy!(probability_matrix(kdst), probability_matrix(ksrc))
    return kdst
end

similar(k::AbstractStochasticMatrix) = similar(k, eltype(k), size(k))
similar(k::AbstractStochasticMatrix, m, n) = similar(k, (m, n))
similar(k::AbstractStochasticMatrix, dims::NTuple{2,IDXT}) where {IDXT} =
    similar(k, eltype(k), dims)
similar(k::AbstractStochasticMatrix, ::Type{T}) where {T} = similar(k, T, size(k))
similar(k::AbstractStochasticMatrix, ::Type{T}, m, n) where {T} = similar(k, T, (m, n))

function similar(
    k::AbstractStochasticMatrix,
    ::Type{T},
    dims::NTuple{2,IDXT},
) where {T,IDXT}
    P = probability_matrix(k)
    Pout = similar(P, T, dims)
    return StochasticMatrix(Pout, false)
end

function Base.show(io::IO, K::AbstractStochasticMatrix)
    println(io, summary(K))
    println(io, "P = ")
    show(io, K.P)
end
