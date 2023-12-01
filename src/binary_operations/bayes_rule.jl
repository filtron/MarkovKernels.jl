"""
    bayes_rule(D::AbstractDistribution, K::AbstractMarkovKernel, y)

Computes the conditional distribution C and the marginal log-likelihood ℓ associated with the prior distribution D, measurement kernel K, and measurement y.
"""
function bayes_rule(D::AbstractDistribution, K::AbstractMarkovKernel, y)
    M, C = invert(D, K)
    return condition(C, y), logpdf(M, y)
end

"""
    bayes_rule(D::AbstractDistribution, L::AbstractLikelihood)

Computes the conditional distribution C and the marginal log-likelihood ℓ associated with the prior distribution D and the log-likelihood L.
"""
bayes_rule(D::AbstractDistribution, L::AbstractLikelihood) =
    bayes_rule(D, measurement_model(L), measurement(L))

#= 
loglike is always Float64, bad?
should be: 

twople_types(::Tuple{A,B}) where {A,B} = A, B 
last(twople_types(first(Base.return_types(logpdf, (typeof(D), sample_type(D))))))

but Base.return_types is internal and sample_type is not implemented 
=#
bayes_rule(D::AbstractDistribution, ::FlatLikelihood) = D, 0.0

function bayes_rule(
    P::ParticleSystem{T,U,<:AbstractVector},
    L::AbstractLikelihood,
) where {T,U}
    logws = copy(logweights(P))
    loglike = _update_weights_and_compute_loglike!(logws, P, L)

    return ParticleSystem(logws, copy.(particles(P))), loglike
end

function bayes_rule(
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
function bayes_rule!(
    P::ParticleSystem{T,U,<:AbstractVector},
    L::AbstractLikelihood,
) where {T,U}
    return _update_weights_and_compute_loglike!(logweights(P), P, L)
end

function bayes_rule!(
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
