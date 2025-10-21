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

"""
    ProbabilityVector(p::AbstractVector)

Constructs a categorical distribution from the vector of probabilities p.
"""
function ProbabilityVector(p::AbstractVector, normalize = true)
    if normalize
        π = copy(p)
        normalize!(π, 1)
    else
        π = p
    end
    return ProbabilityVector{eltype(eachindex(π)),typeof(π)}(π)
end

probability_vector(d::ProbabilityVector) = d.p
noutcomes(d::ProbabilityVector) = length(probability_vector(d))
dim(d::ProbabilityVector) = 1

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
        e_incr = ifelse(iszero(pi), zero(pi), -log(pi) * pi)
        e = e + e_incr
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
    T = promote_type(eltype(p1), eltype(p2))
    eachindex(p1) != eachindex(p2) && return Inf
    kld = zero(float(T))
    for i in eachindex(p1)
        logratio = ifelse(iszero(p1[i]) && iszero(p2[i]), zero(T), log(p1[i]) - log(p2[i]))
        kld = kld + logratio * p1[i]
    end
    return kld
end

function rand(rng::AbstractRNG, C::AbstractProbabilityVector)
    p = probability_vector(C)
    at = AliasTable(p)
    return sample_type(C)(rand(rng, at))
end

function Base.copy!(Cdst::ProbabilityVector, Csrc::ProbabilityVector)
    copy!(probability_vector(Cdst), probability_vector(Csrc))
    return Cdst
end

Base.isapprox(C1::ProbabilityVector, C2::ProbabilityVector, kwargs...) =
    isapprox(probability_vector(C1), probability_vector(C2), kwargs...)

similar(d::ProbabilityVector{ST,VT}) where {ST,VT} = similar(d, eltype(VT), noutcomes(d))
similar(d::ProbabilityVector{ST,VT}, m) where {ST,VT} = similar(d, eltype(VT), m)
similar(d::ProbabilityVector, ::Type{T}) where {T} = similar(d, T, noutcomes(d))

function similar(d::ProbabilityVector, ::Type{T}, m) where {T}
    p = probability_vector(d)
    pout = similar(p, T, m)
    return ProbabilityVector(pout, false)
end

function Base.show(io::IO, C::ProbabilityVector)
    println(io, summary(C))
    print(io, "p = ")
    show(io, probability_vector(C))
end
