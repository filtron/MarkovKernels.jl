"""
    AbstractStochasticMatrix

Abstract type for representing stochastic matrices.
"""
abstract type AbstractStochasticMatrix <: AbstractMarkovKernel end

struct StochasticMatrix{A} <: AbstractStochasticMatrix
    P::A
end

function probability_matrix(K::StochasticMatrix)
    P = copy(K.P)
    foreach(_normalize_vector!, eachcol(P))
    return P
end

"""
    condition(K::AbstractStochasticMatrix, x)

Returns a Dirac distribution corresponding to the Dirac kernel K evaluated at x.
"""
condition(K::AbstractStochasticMatrix, x) = Categorical(K.P[:, x])

rand(rng::AbstractRNG, K::AbstractStochasticMatrix, x::Int) = rand(rng, condition(K, x))
rand(K::AbstractStochasticMatrix, x::Int) = rand(Random.default_rng(), K, x)

function Base.show(io::IO, K::AbstractStochasticMatrix)
    println(io, summary(K))
    println(io, "P = ")
    show(io, K.P)
end
