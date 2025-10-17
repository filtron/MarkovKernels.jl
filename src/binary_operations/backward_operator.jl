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

# apparently needed to break ambiguity 
backward_operator(::Likelihood{<:StochasticMatrix,<:Missing}, k::StochasticMatrix) =
    backward_operator(FlatLikelihood(), k)

function backward_operator(h::Likelihood{<:StochasticMatrix}, k::StochasticMatrix)
    P = probability_matrix(k)
    hout = similar(h, axes(P, 2))
    return backward_operator!(hout, h, k)
end

function backward_operator!(
    hout::LikelihoodVector,
    h::Likelihood{<:StochasticMatrix},
    k::StochasticMatrix,
)
    hk = measurement_model(h)
    hy = measurement(h)
    P = probability_matrix(hk)
    ls = view(P, hy, :)
    h = LikelihoodVector(ls)
    return backward_operator!(hout, h, k)
end

function backward_operator(h::LikelihoodVector, k::StochasticMatrix)
    P = probability_matrix(k)
    hout = similar(h, axes(P, 2))
    hout = backward_operator!(hout, h, k)
    return hout
end

function backward_operator!(
    hout::LikelihoodVector,
    h::LikelihoodVector,
    k::StochasticMatrix,
)
    lsout = likelihood_vector(hout)
    ls = likelihood_vector(h)
    P = probability_matrix(k)
    lsout = mul!(lsout, adjoint(P), ls)
    return hout
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
