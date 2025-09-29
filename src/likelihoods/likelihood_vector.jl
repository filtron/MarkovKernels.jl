"""
    LikelihoodVector

Type for representing a Likelihood function over categories.
"""
struct LikelihoodVector{A} <: AbstractLikelihood
    ls::A
end

"""
    LikelihoodVector(L::Likelihood{<:StochasticMatrix})

Computes a categorical likelihood from L.
"""
function LikelihoodVector(L::Likelihood{<:StochasticMatrix})
    K, y = measurement_model(L), measurement(L)
    P = probability_matrix(K)
    ls = P[y, :]
    return LikelihoodVector{typeof(ls)}(ls)
end

"""
    likelihood_vector(L::LikelihoodVector)

Computes the vector of likelihood evaluations.
"""
likelihood_vector(L::LikelihoodVector) = L.ls

function likelihood_vector(L::Likelihood{<:StochasticMatrix})
    K, y = measurement_model(L), measurement(L)
    P = probability_matrix(K)
    ls = P[y, :]
    return ls
end

log(L::LikelihoodVector, x) = log(likelihood_vector(L)[x])
