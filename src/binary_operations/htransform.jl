"""
    htransform_and_likelihood(K::AbstractMarkovKernel, L::AbstractLikelihood)

Computes a Markov kernel Kout and Likelihood Lout such that

Lout(z) = ∫ L(x) K(x, z) dx, and Kout(x, z) = L(x) K(x, z) / Lout(z)

"""
function htransform_and_likelihood(::AbstractMarkovKernel, ::AbstractLikelihood) end

htransform_and_likelihood(K::AbstractMarkovKernel, L::FlatLikelihood) = K, L
htransform_and_likelihood(
    K::AbstractMarkovKernel,
    ::Likelihood{<:AbstractMarkovKernel,<:Missing},
) = htransform_and_likelihood(K, FlatLikelihood())

function htransform_and_likelihood(K::StochasticMatrix, L::CategoricalLikelihood)
    ls = likelihood_vector(L)
    P = probability_matrix(K)

    lsout = similar(P, size(P, 2))
    lsout = mul!(lsout, adjoint(P), ls)

    Pout = similar(P)

    for i in axes(P, 1), j in axes(P, 2)
        Pout[i, j] = ls[i] * P[i, j] / lsout[j]
    end

    Lout = CategoricalLikelihood(lsout)
    Kout = StochasticMatrix(Pout)
    return Kout, Lout
end

function htransform_and_likelihood(K::StochasticMatrix, L::Likelihood{<:StochasticMatrix})
    Kobs, y = measurement_model(L), measurement(L)
    ls = view(probability_matrix(Kobs), y, :)
    Ltmp = CategoricalLikelihood(ls)
    return htransform_and_likelihood(K, Ltmp)
end

function htransform_and_likelihood(
    K::AffineHomoskedasticNormalKernel{TM,TC},
    L::LogQuadraticLikelihood,
) where {TM,TC<:Union{SelfAdjoint,Number}}
    μ, Q = mean(K), covp(K)
    Φ, u = slope(μ), intercept(μ)
    logc, y, C = L
    T = eltype(y)

    Rhat, Kbar, Qpost = schur_reduce(Q, C, I)

    L = lsqrt(Rhat)
    yout = L \ (y - C * u)
    Cout = L \ C * Φ
    logcout = logc - _nscale(T) * 2 * logdet(L)

    Φpost = (I - Kbar * C) * Φ
    upost = u + Kbar * (y - C * u)
    μpost = AffineMap(Φpost, upost)

    Kout = NormalKernel(μpost, Qpost)
    Lout = LogQuadraticLikelihood(logcout, yout, Cout)

    return Kout, Lout
end

function htransform_and_likelihood(
    K::AffineHomoskedasticNormalKernel{TM,<:Cholesky},
    L::LogQuadraticLikelihood,
) where {TM}
    μ, Q = mean(K), covp(K)
    Φ, u = slope(μ), intercept(μ)
    logc, y, C = L
    T = eltype(y)

    Rhat, Kbar, Qpost = _schur_reduce(Q, C, I)

    L = lsqrt(Rhat)
    yout = L \ (y - C * u)
    Cout = L \ C * Φ
    logcout = logc - _nscale(T) * 2 * logdet(L)

    Φpost = Φ - Kbar * Cout
    upost = u + Kbar * yout
    μpost = AffineMap(Φpost, upost)

    Kout = NormalKernel(μpost, Qpost)
    Lout = LogQuadraticLikelihood(logcout, yout, Cout)

    return Kout, Lout
end

htransform_and_likelihood(
    K::AffineHomoskedasticNormalKernel,
    L::Likelihood{<:AffineHomoskedasticNormalKernel},
) = htransform_and_likelihood(K, LogQuadraticLikelihood(L))

function htransform_and_likelihood(
    K::AffineHomoskedasticNormalKernel,
    L::Likelihood{<:AffineDiracKernel},
)
    μ, Q = mean(K), covp(K)
    Φ, u = slope(μ), intercept(μ)

    KL = L.K
    yL = L.y
    C = slope(mean(KL))
    y = yL - intercept(mean(KL))

    Rhat, Kbar, Qpost = schur_reduce(Q, C)

    KLout = NormalKernel(compose(mean(KL), mean(K)), Rhat)
    Lout = Likelihood(KLout, yL) # maybe return LogQuadraticLikelihood for consistency?

    Φpost = (I - Kbar * C) * Φ
    upost = u + Kbar * (y - C * u)
    μpost = AffineMap(Φpost, upost)
    Kout = NormalKernel(μpost, Qpost)

    return Kout, Lout
end
