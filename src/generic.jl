abstract type AbstractDistribution{T<:Number} end

"""
    rand([rng], D::AbstractDistribution)

Draws one sample from D.
"""
function Random.rand(::AbstractRNG, ::AbstractDistribution) end

Random.rand(D::AbstractDistribution) = rand(Random.default_rng(), D)

"""
    sample_type(D::AbstractDistribution)

Computes the type of samples from D, e.g. same as typeof(rand(D)).
"""
function sample_type(::AbstractDistribution) end

"""
eltype_sample(D::AbstractDistribution)

Computes the eltype of samples from D, e.g. same as eltype(rand(D)).
"""
sample_eltype(D::AbstractDistribution) = eltype(sample_type(D))

"""
    logpdf(D::AbstractDistribution, x)

Computes the logarithm of the probabilidty density of D, evaluated at x.
"""
function logpdf(::AbstractDistribution, x) end

abstract type AbstractMarkovKernel end

"""
    condition(K::AbstractMarkovKernel, x)

Computes the distribution retrieved from evaluating K at x.
"""
function condition(::AbstractMarkovKernel, ::Any) end

abstract type AbstractLikelihood end

for T in (:AbstractDistribution, :AbstractMarkovKernel, :AbstractLikelihood)
    for func in (:(==), :isequal, :isapprox)
        @eval function Base.$func(P1::U, P2::V; kwargs...) where {U<:$T,V<:$T}
            nameof(U) === nameof(V) || return false
            fields = fieldnames(U)
            fields === fieldnames(V) || return false

            for f in fields
                isdefined(P1, f) && isdefined(P2, f) || return false
                getfield(P1, f) === getfield(P2, f) ||
                    $func(getfield(P1, f), getfield(P2, f); kwargs...) ||
                    return false
            end

            return true
        end
    end

    for func in (:similar, :copy)
        @eval function Base.$func(F::$T)
            fields = fieldnames(typeof(F))
            input = Tuple($func(getfield(F, f)) for f in fields)
            return typeof(F)(input...)
        end
    end

    @eval function Base.$:copy!(Fdst::A, Fsrc::A) where {A<:$T}
        fields = fieldnames(A)
        dst = Tuple(getfield(Fdst, f) for f in fields)
        src = Tuple(getfield(Fsrc, f) for f in fields)
        foreach(Base.splat(copy!), zip(dst, src))
        return Fdst
    end
end
