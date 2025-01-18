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
    C, _ = posterior_and_loglike(D, K, y)
    return C
end

"""
    posterior_and_loglike(D::AbstractDistribution, L::AbstractLikelihood)

Computes the conditional distribution C and the marginal log-likelihood ℓ associated with the prior distribution D and the likelihood L.
"""
function posterior_and_loglike(::AbstractDistribution, ::AbstractLikelihood) end

"""
    posterior(D::AbstractDistribution, L::AbstractLikelihood)

Computes the conditional distribution C associated with the prior distribution D and the likelihood L.
"""
function posterior(D::AbstractDistribution, L::AbstractLikelihood)
    C, _ = posterior_and_loglike(D, L)
    return C
end

posterior_and_loglike(D::AbstractDistribution, L::Likelihood) =
    posterior_and_loglike(D, measurement_model(L), measurement(L))

function posterior_and_loglike(D::AbstractDistribution, L::LogQuadraticLikelihood)
    logc, y, C = L
    T = eltype(y)
    K = NormalKernel(LinearMap(C), I)
    logc = logc + _nscale(T) * length(y) * _logpiconst(T)
    M, C = invert(D, K)
    return condition(C, y), logc + logpdf(M, y)
end

function posterior_and_loglike(C::Categorical, L::CategoricalLikelihood)
    π = probability_vector(C)
    ls = likelihood_vector(L)

    my = dot(ls, π)
    πout = similar(π)
    πout .= ls .* π ./ my
    Cout = Categorical(πout)

    return Cout, log(my)
end

function posterior_and_loglike(C::Categorical, L::Likelihood{<:StochasticMatrix,<:Int})
    K, y = measurement_model(L), measurement(L)
    P = probability_matrix(K)
    ls = view(P, y, :)
    LC = CategoricalLikelihood(ls)
    return posterior_and_loglike(C, LC)
end

posterior_and_loglike(D::AbstractDistribution, ::FlatLikelihood) = D, 0

posterior_and_loglike(
    D::AbstractDistribution,
    ::Likelihood{<:AbstractMarkovKernel,<:Missing},
) = posterior_and_loglike(D, FlatLikelihood())
