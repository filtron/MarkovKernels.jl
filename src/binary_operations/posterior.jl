"""
    posterior_and_loglike(D::AbstractDistribution, K::AbstractMarkovKernel, y)

Computes the conditional distribution C and the marginal log-likelihood ℓ associated with the prior distribution D, measurement kernel K, and measurement y.
"""
function posterior_and_loglike(D::AbstractDistribution, K::AbstractMarkovKernel, y)
    M, C = invert(D, K)
    return condition(C, y), logpdf(M, y)
end

"""
    posterior(D::AbstractDistribution, K::AbstractMarkovKernel, y)

Computes the conditional distribution C associated with the prior distribution D, measurement kernel K, and measurement y.
"""
function posterior(D::AbstractDistribution, K::AbstractMarkovKernel, y)
    _, C = invert(D, K)
    return condition(C, y)
end

"""
    posterior_and_loglike(D::AbstractDistribution, L::AbstractLikelihood)

Computes the conditional distribution C and the marginal log-likelihood ℓ associated with the prior distribution D and the log-likelihood L.
"""
posterior_and_loglike(D::AbstractDistribution, L::AbstractLikelihood) =
    posterior_and_loglike(D, measurement_model(L), measurement(L))

"""
    posterior(D::AbstractDistribution, L::AbstractLikelihood)

Computes the conditional distribution C associated with the prior distribution D and the log-likelihood L.
"""
posterior(D::AbstractDistribution, L::AbstractLikelihood) =
    posterior(D, measurement_model(L), measurement(L))

posterior_and_loglike(D::AbstractDistribution, ::FlatLikelihood) = D, 0
posterior(D::AbstractDistribution, ::FlatLikelihood) = D
