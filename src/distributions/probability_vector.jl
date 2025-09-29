"""
    AbstractProbabilityVector{ST}

Abstract type for representing categorical distributions with values ST.
"""
abstract type AbstractProbabilityVector{ST} <: AbstractDistribution{ST} end

"""
    probability_vector(::AbstractProbabilityVector)


Computes the vector of probabilities for each category.
"""
function probability_vector(::AbstractProbabilityVector) end

"""
    ProbabilityVector{T,A}


Type for representing categorical distributions with sample_eltype T.
"""
struct ProbabilityVector{T,A} <: AbstractProbabilityVector{T}
    p::A
end

_normalize_vector!(v::AbstractVector) = ldiv!(sum(v), v)

"""
    ProbabilityVector(p::AbstractVector)

Constructs a categorical distribution from the vector of probabilities p.
"""
function ProbabilityVector(p::AbstractVector)
    π = copy(p)
    _normalize_vector!(π)
    return ProbabilityVector{eltype(eachindex(π)),typeof(π)}(π)
end

probability_vector(C::ProbabilityVector) = C.p

dim(C::ProbabilityVector) = 1

function Base.copy!(Cdst::ProbabilityVector, Csrc::ProbabilityVector)
    copy!(probability_vector(Cdst), probability_vector(Csrc))
    return Cdst
end

Base.similar(C::ProbabilityVector) = ProbabilityVector(similar(probability_vector(C)))
Base.isapprox(C1::ProbabilityVector, C2::ProbabilityVector, kwargs...) =
    isapprox(probability_vector(C1), probability_vector(C2), kwargs...)

function logpdf(C::ProbabilityVector, x)
    p = probability_vector(C)
    return log(p[x])
end

"""
    entropy(C::AbstractProbabilityVector)

Computes the entropy of the categorical distribution C.
"""
function entropy(C::AbstractProbabilityVector)
    p = probability_vector(C)
    e = zero(float(eltype(p)))
    for i in eachindex(p)
        pi = p[i]
        e = e - log(pi) * pi
    end
    return e
end

"""
    kldivergence(C1::AbstractProbabilityVector, C2::AbstractProbabilityVector)

Computes the Kullback-Leibler divergence between the categorical distributions C1 and C2.
"""
function kldivergence(C1::AbstractProbabilityVector, C2::AbstractProbabilityVector)
    p1 = probability_vector(C1)
    p2 = probability_vector(C2)
    eachindex(p1) != eachindex(p2) && return Inf
    kld = zero(float(eltype(p1)))
    for i in eachindex(p1)
        logratio = log(p1[i]) - log(p2[i])
        kld = kld + logratio * p1[i]
    end
    return kld
end

function rand(rng::AbstractRNG, C::AbstractProbabilityVector)
    p = probability_vector(C)
    at = AliasTable(p)
    return sample_type(C)(rand(rng, at))
end

function Base.show(io::IO, C::ProbabilityVector)
    println(io, summary(C))
    print(io, "p = ")
    show(io, probability_vector(C))
end
