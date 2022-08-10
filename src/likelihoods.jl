abstract type AbstractLikelihood end

struct Likelihood{U<:AbstractMarkovKernel,V} <: AbstractLikelihood
    K::U
    y::V
end

measurement_model(L::Likelihood) = L.K
measurement(L::Likelihood) = L.y
