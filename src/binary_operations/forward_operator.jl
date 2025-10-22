"""
    forward_operator(k::AbstractMarkovKernel, d)

Computes the output of the forward operator associated with k, gvien the input d, i.e.

```math
∫ k(y, x) d(x) dx
```
"""
function forward_operator(k::AbstractMarkovKernel, d) end

forward_operator(k::AffineHomoskedasticNormalKernel, d::AbstractNormal) =
    Normal(mean(k)(mean(d)), stein(covparam(d), mean(k), covparam(k)))

forward_operator(k::AffineDiracKernel, d::AbstractNormal) =
    Normal(mean(k)(mean(d)), stein(covparam(d), mean(k)))

function forward_operator(
    k::AffineDiracKernel,
    d::CholeskyNormal,
    work_arr = similar(slope(mean(k)), reverse(size(slope(mean(k))))),
)
    a = mean(k)
    m, n = size(slope(a))
    dout = similar(d, m)
    return forward_operator!(dout, k, d)
end

function forward_operator!(
    dout::CholeskyNormal,
    k::AffineDiracKernel,
    d::CholeskyNormal,
    work_arr = similar(slope(mean(k)), reverse(size(slope(mean(k))))),
)
    μout, Σout = mean(dout), covparam(dout)
    a = mean(k)
    μ, Σ = mean_and_covparam(d)
    a(μout, μ)
    stein!(Σout, Σ, slope(a), work_arr)
    return dout
end

forward_operator(k::AbstractMarkovKernel, d::AbstractDirac) = condition(k, mean(d))

function forward_operator(k::StochasticMatrix, d::AbstractProbabilityVector)
    P = probability_matrix(k)
    dout = similar(d, size(P, 1))
    return forward_operator!(dout, k, d)
end

function forward_operator!(
    dout::ProbabilityVector,
    k::StochasticMatrix,
    d::AbstractProbabilityVector,
)
    π = probability_vector(d)
    πout = probability_vector(dout)
    P = probability_matrix(k)
    mul!(πout, P, π)
    return dout
end

forward_operator(::IdentityKernel, d::AbstractDistribution) = d
forward_operator(::IdentityKernel, d::AbstractDirac) = d

forward_operator(k2::AffineHomoskedasticNormalKernel, k1::AffineHomoskedasticNormalKernel) =
    NormalKernel(compose(mean(k2), mean(k1)), stein(covparam(k1), mean(k2), covparam(k2)))

forward_operator(k2::AffineHomoskedasticNormalKernel, k1::AffineDiracKernel) =
    NormalKernel(compose(mean(k2), mean(k1)), covparam(k2))

forward_operator(k2::AffineHeteroskedasticNormalKernel, k1::AffineDiracKernel) =
    NormalKernel(compose(mean(k2), mean(k1)), covparam(k2) ∘ mean(k1))

forward_operator(k2::AffineDiracKernel, k1::AffineDiracKernel) =
    DiracKernel(compose(mean(k2), mean(k1)))

forward_operator(k2::AffineDiracKernel, k1::AffineHomoskedasticNormalKernel) =
    NormalKernel(compose(mean(k2), mean(k1)), stein(covparam(k1), mean(k2)))

function forward_operator(k2::StochasticMatrix, k1::StochasticMatrix)
    P2 = probability_matrix(k2)
    P1 = probability_matrix(k1)
    m, n = size(P2, 1), size(P1, 2)
    P3 = similar(P2, m, n)
    mul!(P3, P2, P1)
    return StochasticMatrix(P3)
end

forward_operator(k2::AbstractMarkovKernel, ::IdentityKernel) = k2
forward_operator(::IdentityKernel, k1::AbstractMarkovKernel) = k1
forward_operator(k2::IdentityKernel, ::IdentityKernel) = k2
