==(L1::T, L2::T) where {T<:AbstractLogLike} =
    all(f -> getfield(L1, f) == getfield(L2, f), 1:nfields(L1))

struct LogLike{U,V} <: AbstractLogLike
    K::U
    y::V
    LogLike{U,V}(K, y) where {U,V} = new{U,V}(K, y)
end

LogLike(K::AbstractMarkovKernel, y) = LogLike{typeof(K),typeof(y)}(K, y)

measurement_model(L::LogLike) = L.K
measurement(L::LogLike) = L.y

(L::LogLike)(x) = logpdf(condition(measurement_model(L), x), measurement(L))

function bayes_rule(D::AbstractDistribution, K::AbstractMarkovKernel, y)
    M, C = invert(D, K)
    return condition(C, y), logpdf(M, y)
end

bayes_rule(D::AbstractDistribution, L::AbstractLogLike) =
    bayes_rule(D, measurement_model(L), measurement(L))
