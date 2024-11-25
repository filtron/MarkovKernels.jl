"""
    AbstractMarkovKernel

Abstract type for representing Markov kernels.
"""
abstract type AbstractMarkovKernel end

"""
    condition(K::AbstractMarkovKernel, x)

Computes the distribution retrieved from evaluating K at x.
"""
function condition(::AbstractMarkovKernel, ::Any) end

for func in (:(==), :isequal, :isapprox)
    @eval function Base.$func(K1::AbstractMarkovKernel, K2::AbstractMarkovKernel; kwargs...)
        U, V = typeof(K1), typeof(K2)
        nameof(U) === nameof(V) || return false
        fields = fieldnames(U)
        fields === fieldnames(V) || return false

        for f in fields
            isdefined(K1, f) && isdefined(K2, f) || return false
            getfield(K1, f) === getfield(K2, f) ||
                $func(getfield(K1, f), getfield(K2, f); kwargs...) ||
                return false
        end

        return true
    end
end

for func in (:similar, :copy)
    @eval function Base.$func(K::AbstractMarkovKernel)
        fields = fieldnames(typeof(K))
        input = Tuple($func(getfield(K, f)) for f in fields)
        return typeof(K)(input...)
    end
end

function Base.copy!(Kdst::A, Ksrc::A) where {A<:AbstractMarkovKernel}
    fields = fieldnames(A)
    dst = Tuple(getfield(Kdst, f) for f in fields)
    src = Tuple(getfield(Ksrc, f) for f in fields)
    foreach(Base.splat(copy!), zip(dst, src))
    return Kdst
end
