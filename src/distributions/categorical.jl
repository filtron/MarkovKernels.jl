
abstract type AbstractCategorical{T} <: AbstractDistribution{T} end

function probability_vector(::AbstractCategorical) end

struct Categorical{T,A} <: AbstractCategorical{T}
    p::A
end

Categorical(p::AbstractVector) = Categorical{eltype(eachindex(p)),typeof(p)}(p)

probability_vector(C::Categorical) = C.p

dim(C::Categorical) = 1

sample_type(C::Categorical) = eltype(eachindex(probability_vector(C)))

function Base.copy!(Cdst::Categorical, Csrc::Categorical)
    copy!(probability_vector(Cdst), probability_vector(Csrc))
    return Cdst
end

Base.similar(C::Categorical) = Categorical(similar(probability_vector(C)))
Base.isapprox(C1::Categorical, C2::Categorical, kwargs...) =
    isapprox(probability_vector(C1), probability_vector(C2), kwargs...)

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
