"""
    AbstractCategorical{T}

Abstract type for representing categorical distributions with values T.
"""
abstract type AbstractCategorical{T} <: AbstractDistribution{T} end

"""
    probability_vector(::AbstractCategorical)


Computes the vector of probabilities for each category. 
"""
function probability_vector(::AbstractCategorical) end

"""
    Categorical{T,A}


Type for representing categorical distributions with sample_eltype T.
"""
struct Categorical{T,A} <: AbstractCategorical{T}
    p::A
end

"""
    Categorical(p::AbstractVector)

Constructs a categorical distribution from the vector of probabilities p. 
"""
Categorical(p::AbstractVector) = Categorical{eltype(eachindex(p)),typeof(p)}(p)

probability_vector(C::Categorical) = C.p / sum(C.p)

dim(C::Categorical) = 1

sample_type(C::Categorical) = eltype(eachindex(probability_vector(C)))

function Base.copy!(Cdst::Categorical, Csrc::Categorical)
    copy!(probability_vector(Cdst), probability_vector(Csrc))
    return Cdst
end

Base.similar(C::Categorical) = Categorical(similar(probability_vector(C)))
Base.isapprox(C1::Categorical, C2::Categorical, kwargs...) =
    isapprox(probability_vector(C1), probability_vector(C2), kwargs...)

function logpdf(C::Categorical, x)
    p = probability_vector(C)
    return log(p[x])
end

function entropy(C::Categorical)
    p = probability_vector(C)
    e = zero(float(eltype(p)))
    for i in eachindex(p)
        pi = p[i]
        e = e - log(pi) * pi
    end
    return e
end

function kldivergence(C1::Categorical, C2::Categorical)
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

function rand(::AbstractRNG, C::AbstractCategorical)
    p = probability_vector(C)
    at = AliasTable(p)
    return rand(at)
end

function Base.show(io::IO, C::Categorical)
    println(io, summary(C))
    print(io, "p = ")
    show(io, probability_vector(C))
end
