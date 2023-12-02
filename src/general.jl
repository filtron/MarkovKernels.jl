abstract type AbstractDistribution{T<:Number} end

eltype(::AbstractDistribution{T}) where {T} = T

abstract type AbstractMarkovKernel{T<:Number} end

eltype(::AbstractMarkovKernel{T}) where {T} = T

abstract type AbstractLikelihood end

for T in
    (:AbstractAffineMap, :AbstractDistribution, :AbstractMarkovKernel, :AbstractLikelihood)
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
        foreach(splat(copy!), zip(dst, src))
        return Fdst
    end
end
