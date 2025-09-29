"""
    forward_operator(k::AbstractMarkovKernel, d::AbstractDistribution)

Computes the output of the forward operator associated with k, givne the input d, i.e.

```math
∫ k(y, x) d(x) dx
```
"""
function forward_operator(k::AbstractMarkovKernel, d::AbstractDistribution) end

forward_operator(k::AffineHomoskedasticNormalKernel, d::AbstractNormal) =
    Normal(mean(k)(mean(d)), stein(covp(d), mean(k), covp(k)))
forward_operator(k::AffineDiracKernel, d::AbstractNormal) =
    Normal(mean(k)(mean(d)), stein(covp(d), mean(k)))
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
