abstract type AbstractLikelihood end

struct Likelihood{U<:AbstractMarkovKernel,V} <: AbstractLikelihood
    K::U
    y::V
end

measurement_model(L::Likelihood) = L.K
measurement(L::Likelihood) = L.y

function bayes_rule(D::AbstractDistribution, y, K::AbstractMarkovKernel)
    M, C = invert(D, K)
    return condition(C, y), logpdf(M, y)
end

bayes_rule(D::AbstractDistribution, L::AbstractLikelihood) =
    bayes_rule(D, measurement(L), measurement_model(L))
