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
forward_operator(k::AbstractMarkovKernel, d::AbstractDirac) = condition(k, mean(d))

function forward_operator(k::StochasticMatrix, d::AbstractCategorical)
    π = probability_vector(d)
    P = probability_matrix(k)
    πout = similar(π, size(P, 1))
    mul!(πout, P, π)
    return Categorical(πout)
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
