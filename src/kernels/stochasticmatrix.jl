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

function _normalize_matrix!(P::AbstractMatrix)
    foreach(_normalize_vector!, eachcol(P))
end

"""
StochasticMatrix(P::AbstractMatrix)

Constructs a stochastic matrix from the matrix of transition probabilities P.
"""
function StochasticMatrix(P::AbstractMatrix)
    Π = copy(P)
    _normalize_matrix!(Π)
    return StochasticMatrix{typeof(Π)}(Π)
end

probability_matrix(K::StochasticMatrix) = K.P

condition(K::AbstractStochasticMatrix, x) = Categorical(K.P[:, x])

"""
    rand([rng::AbstractRNG], K::AbstractStochasticMatrix, x)

Samples a random vector conditionally on x with respect the the stochastic matrix K
using the random number generator rng.
"""
rand(rng::AbstractRNG, K::AbstractStochasticMatrix, x::Int) = rand(rng, condition(K, x))
rand(K::AbstractStochasticMatrix, x::Int) = rand(Random.default_rng(), K, x)

function Base.show(io::IO, K::AbstractStochasticMatrix)
    println(io, summary(K))
    println(io, "P = ")
    show(io, K.P)
end
