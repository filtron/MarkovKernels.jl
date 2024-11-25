"""
    AbstractLikelihood

Abstract type for representing likelihood functions.
"""
abstract type AbstractLikelihood end

"""
    log(L::AbstractLikelihood, x)

Computes the logarithm of L evaluated at x.
"""
function log(::AbstractLikelihood, ::Any) end

for func in (:(==), :isequal, :isapprox)
    @eval function Base.$func(L1::AbstractLikelihood, L2::AbstractLikelihood; kwargs...)
        U, V = typeof(L1), typeof(L2)
        nameof(U) === nameof(V) || return false
        fields = fieldnames(U)
        fields === fieldnames(V) || return false

        for f in fields
            isdefined(L1, f) && isdefined(L2, f) || return false
            getfield(L1, f) === getfield(L2, f) ||
                $func(getfield(L1, f), getfield(L2, f); kwargs...) ||
                return false
        end

        return true
    end
end

for func in (:similar, :copy)
    @eval function Base.$func(L::AbstractLikelihood)
        fields = fieldnames(typeof(L))
        input = Tuple($func(getfield(L, f)) for f in fields)
        return typeof(L)(input...)
    end
end

function Base.copy!(Ldst::A, Lsrc::A) where {A<:AbstractLikelihood}
    fields = fieldnames(A)
    dst = Tuple(getfield(Ldst, f) for f in fields)
    src = Tuple(getfield(Lsrc, f) for f in fields)
    foreach(Base.splat(copy!), zip(dst, src))
    return Ldst
end
