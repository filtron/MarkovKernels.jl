"""
    CategoricalLikelihood

Type for representing a Likelihood function over categories.
"""
struct CategoricalLikelihood{A} <: AbstractLikelihood
    ls::A
end

function CategoricalLikelihood(L::Likelihood{<:StochasticMatrix})
    K, y = measurement_model(L), measurement(L)
    P = probability_matrix(K)
    ls = P[y, :]
    return CategoricalLikelihood{typeof(ls)}(ls)
end

likelihood_vector(L::CategoricalLikelihood) = L.ls
log(L::CategoricalLikelihood, x) = log(likelihood_vector(L)[x])
