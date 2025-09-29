"""
    backward_operator(h, k::AbstractMarkovKernel)

Computes the output of the backward operator associated with k, gvien the input h, i.e.

```math
∫ h(y) k(y, x) dy
```
"""
function backward_operator(h, k::AbstractMarkovKernel) end

backward_operator(k1::AbstractMarkovKernel, k2::AbstractMarkovKernel) =
    forward_operator(k2, k1)
backward_operator(::FlatLikelihood, k::AbstractMarkovKernel) = FlatLikelihood()
backward_operator(::Likelihood{<:AbstractMarkovKernel,<:Missing}, k::AbstractMarkovKernel) =
    backward_operator(FlatLikelihood(), k)

function backward_operator(h::LikelihoodVector, k::StochasticMatrix)
    ls = likelihood_vector(h)
    P = probability_matrix(k)

    lsout = similar(P, axes(P, 2))
    lsout = mul!(lsout, adjoint(P), ls)
    hout = LikelihoodVector(lsout)
    return hout
end

function backward_operator(h::Likelihood{<:StochasticMatrix}, k::StochasticMatrix)
    Kobs, y = measurement_model(h), measurement(h)
    ls = view(probability_matrix(Kobs), y, :)
    htmp = LikelihoodVector(ls)
    return backward_operator(htmp, k)
end

function backward_operator(h::LogQuadraticLikelihood, k::AffineHomoskedasticNormalKernel)
    μ, Q = mean(k), covparam(k)
    Φ, u = slope(μ), intercept(μ)
    logc, y, C = h
    T = eltype(y)

    Rhat = stein(Q, C, I)

    L = lsqrt(Rhat)
    yout = L \ (y - C * u)
    Cout = L \ C * Φ
    logcout = logc - _nscale(T) * 2 * logdet(L)

    hout = LogQuadraticLikelihood(logcout, yout, Cout)
    return hout
end

backward_operator(
    h::Likelihood{<:AffineHomoskedasticNormalKernel},
    k::AffineHomoskedasticNormalKernel,
) = backward_operator(LogQuadraticLikelihood(h), k)

function backward_operator(
    h::Likelihood{<:AffineDiracKernel},
    k::AffineHomoskedasticNormalKernel,
)
    μ, Q = mean(k), covparam(k)
    Φ, u = slope(μ), intercept(μ)

    kh = h.K
    yh = h.y
    C = slope(mean(kh))
    yhout = yh - intercept(mean(kh))

    Rhat = stein(Q, C)

    Khout = NormalKernel(compose(mean(kh), mean(k)), Rhat)
    hout = Likelihood(Khout, yhout) # maybe return LogQuadraticLikelihood for consistency?

    return hout
end
