"""
    AbstractLikelihood{T<:Number} 

Abstract type for representing likelihoods 
"""
abstract type AbstractLogLike{T<:Number} end

eltype(::AbstractLogLike{T}) where {T} = T

AbstractLogLike{T}(L::AbstractLogLike{T}) where {T} = L

==(L1::T, L2::T) where {T<:AbstractLogLike} =
    all(f -> getfield(L1, f) == getfield(L2, f), 1:nfields(L1))

struct LogLike{T,U,V} <: AbstractLogLike{T}
    K::U
    y::V
    LogLike{T}(K, y) where {T} = new{T,typeof(K),typeof(y)}(K, y)
end

function LogLike(K::AbstractMarkovKernel{T}, y::AbstractVector{T}) where {T}
    # insert conversion
    LogLike{T}(K, y)
end

measurement_model(L::LogLike) = L.K
measurement(L::LogLike) = L.y

(L::LogLike)(x) = logpdf(condition(measurement_model(L), x), measurement(L))

function bayes_rule(D::AbstractDistribution, K::AbstractMarkovKernel, y)
    M, C = invert(D, K)
    return condition(C, y), logpdf(M, y)
end

bayes_rule(D::AbstractDistribution, L::AbstractLogLike) =
    bayes_rule(D, measurement_model(L), measurement(L))
