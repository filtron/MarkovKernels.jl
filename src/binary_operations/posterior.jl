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

posterior_and_loglike(
    D::ParticleSystem{T,U,<:AbstractVector},
    ::FlatLikelihood,
) where {T,U} = D, 0

function posterior_and_loglike(
    P::ParticleSystem{T,U,<:AbstractVector},
    L::AbstractLikelihood,
) where {T,U}
    logws = copy(logweights(P))
    loglike = _update_weights_and_compute_loglike!(logws, P, L)

    return ParticleSystem(logws, copy.(particles(P))), loglike
end

posterior_and_loglike(
    D::ParticleSystem{T,U,<:AbstractMatrix},
    ::FlatLikelihood,
) where {T,U} = D, 0

function posterior_and_loglike(
    P::ParticleSystem{T,U,<:AbstractMatrix},
    L::AbstractLikelihood,
) where {T,U}
    latest_time_marginal = ParticleSystem(logweights(P), particles(P)[end, :])
    logws = copy(logweights(latest_time_marginal))
    loglike = _update_weights_and_compute_loglike!(logws, latest_time_marginal, L)
    return ParticleSystem(logws, copy.(particles(P))), loglike
end

"""
    bayes_rule!(D::AbstractParticleSystem, L::AbstractLikelihood)

Computes the conditional distribution C in-place and the marginal log-likelihood ℓ associated with the prior distribution D and the log-likelihood L.
"""
function posterior_and_loglike!(
    P::ParticleSystem{T,U,<:AbstractVector},
    L::AbstractLikelihood,
) where {T,U}
    return _update_weights_and_compute_loglike!(logweights(P), P, L)
end

function posterior_and_loglike!(
    P::ParticleSystem{T,U,<:AbstractMatrix},
    L::AbstractLikelihood,
) where {T,U}
    latest_time_marginal = ParticleSystem(logweights(P), particles(P)[end, :])
    return _update_weights_and_compute_loglike!(
        logweights(latest_time_marginal),
        latest_time_marginal,
        L,
    )
end

function _update_weights_and_compute_loglike!(
    logws::U,
    P::ParticleSystem{T,U,<:AbstractVector},
    L::AbstractLikelihood,
) where {T,U}
    logc1 = maximum(logws)
    logs1 = log(sum(exp, logws .- logc1))

    logws[:] .= logws + [log(L, particles(P)[i]) for i in eachindex(particles(P))]
    logc2 = maximum(logws)

    logws[:] = logws .- logc2
    logs2 = log(sum(exp, logws))

    return logc2 - logc1 + logs2 - logs1
end

const bayes_rule_and_loglike = posterior_and_loglike
const bayes_rule_and_loglike! = posterior_and_loglike!
const bayes_rule = posterior
