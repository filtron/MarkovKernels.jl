"""
    LogLike{U,V}

Type for representing a log-likelihood associated with a kernel K(y, x) and a measurement y.
"""
struct LogLike{U,V} <: AbstractLogLike
    K::U
    y::V
    LogLike{U,V}(K, y) where {U,V} = new{U,V}(K, y)
end

"""
    LogLike(K::AbstractMarkovKernel, y)

Creates a LogLike with measurement kernel K and measurement y.
"""
LogLike(K::AbstractMarkovKernel, y) = LogLike{typeof(K),typeof(y)}(K, y)

"""
    measurement_model(L::LogLike)

Computes the measurement kernel K.
"""
measurement_model(L::LogLike) = L.K

"""
    measurement(L::LogLike)

Computes the measurement y
"""
measurement(L::LogLike) = L.y

"""
    (L::AbstractLogLike)(x)

Computes the log-likelihood associated with L evaluated at x.
"""
(L::LogLike)(x) = logpdf(condition(measurement_model(L), x), measurement(L))
