"""
    bayes_rule(D::AbstractDistribution, K::AbstractMarkovKernel, y)

Computes the conditional distribution C and the marginal log-likelihood ℓ associated with the prior distribution D, measurement kernel K, and measurement y.
"""
function bayes_rule(D::AbstractDistribution, K::AbstractMarkovKernel, y)
    M, C = invert(D, K)
    return condition(C, y), logpdf(M, y)
end

"""
    bayes_rule(D::AbstractDistribution, L::AbstractLogLike)

Computes the conditional distribution C and the marginal log-likelihood ℓ associated with the prior distribution D and the log-likelihood L.
"""
bayes_rule(D::AbstractDistribution, L::AbstractLogLike) =
    bayes_rule(D, measurement_model(L), measurement(L))

function bayes_rule(P::ParticleSystem{T,U,<:AbstractVector}, L::AbstractLogLike) where {T,U}
    logws = copy(logweights(P))
    loglike = _update_weights_and_coompute_loglike!(logws, P, L)

    return ParticleSystem(logws, copy.(particles(P))), loglike
end

function bayes_rule!(
    P::ParticleSystem{T,U,<:AbstractVector},
    L::AbstractLogLike,
) where {T,U}
    return _update_weights_and_coompute_loglike!(logweights(P), P, L)
end

function _update_weights_and_coompute_loglike!(
    logws::U,
    P::ParticleSystem{T,U,<:AbstractVector},
    L::AbstractLogLike,
) where {T,U}
    logc1 = maximum(logws)
    logs1 = log(sum(exp, logws .- logc1))

    logws[:] .= logws + L.(particles(P))
    logc2 = maximum(logws)
    logws[:] = logws .- logc2
    logs2 = log(sum(exp, logws))

    return logc2 - logc1 + logs2 - logs1
end
