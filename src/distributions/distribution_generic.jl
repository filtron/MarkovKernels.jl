"""
AbstractDistribution{ST}

Abstract type for representing distributions with samples of type ST.
"""
abstract type AbstractDistribution{ST} end

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
sample_type(::AbstractDistribution{ST}) where {ST} = ST

"""
sample_eltype(D::AbstractDistribution)

Computes the eltype of samples from D, e.g. same as eltype(rand(D)).
"""
sample_eltype(D::AbstractDistribution) = eltype(sample_type(D))

"""
    logpdf(D::AbstractDistribution, x)

Computes the logarithm of the probabilidty density of D, evaluated at x.
"""
function logpdf(::AbstractDistribution, x) end

for func in (:(==), :isequal, :isapprox)
    @eval function Base.$func(D1::AbstractDistribution, D2::AbstractDistribution; kwargs...)
        U, V = typeof(D1), typeof(D2)
        nameof(U) === nameof(V) || return false
        fields = fieldnames(U)
        fields === fieldnames(V) || return false

        for f in fields
            isdefined(D1, f) && isdefined(D2, f) || return false
            getfield(D1, f) === getfield(D2, f) ||
                $func(getfield(D1, f), getfield(D2, f); kwargs...) ||
                return false
        end

        return true
    end
end

for func in (:similar, :copy)
    @eval function Base.$func(D::AbstractDistribution)
        fields = fieldnames(typeof(D))
        input = Tuple($func(getfield(D, f)) for f in fields)
        return typeof(D)(input...)
    end
end

function Base.copy!(Ddst::A, Dsrc::A) where {A<:AbstractDistribution}
    fields = fieldnames(A)
    dst = Tuple(getfield(Ddst, f) for f in fields)
    src = Tuple(getfield(Dsrc, f) for f in fields)
    foreach(Base.splat(copy!), zip(dst, src))
    return Ddst
end
