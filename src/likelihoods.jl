"""
    Likelihood{U,V}

Type for representing a log-likelihood associated with a kernel K(y, x) and a measurement y.
"""
struct Likelihood{U,V} <: AbstractLikelihood
    K::U
    y::V
    Likelihood{U,V}(K, y) where {U,V} = new{U,V}(K, y)
end

"""
    Likelihood(K::AbstractMarkovKernel, y)

Creates a Likelihood with measurement kernel K and measurement y.
"""
Likelihood(K::AbstractMarkovKernel, y) = Likelihood{typeof(K),typeof(y)}(K, y)

"""
    measurement_model(L::Likelihood)

Computes the measurement kernel K.
"""
measurement_model(L::Likelihood) = L.K

"""
    measurement(L::Likelihood)

Computes the measurement y
"""
measurement(L::Likelihood) = L.y

"""
    (L::AbstractLikelihood)(x)

Computes the log-likelihood associated with L evaluated at x.
"""
(L::Likelihood)(x) = logpdf(condition(measurement_model(L), x), measurement(L))
