"""
    marginalise(D::AbstractDistribution, K::AbstractMarkovKernel)

Computes M, the marginalization of K with respect to D, i.e.,

M(y) = ∫ K(y,x)D(x) dx
"""
marginalize(N::AbstractNormal, K::AffineHomoskedasticNormalKernel) =
    Normal(mean(K)(mean(N)), stein(covp(N), mean(K), covp(K)))

marginalize(N::AbstractNormal, K::AffineDiracKernel) =
    Normal(mean(K)(mean(N)), stein(covp(N), mean(K)))

marginalize(D::AbstractDirac, K::AbstractMarkovKernel) = condition(K, mean(D))

function marginalize(D::Categorical, K::StochasticMatrix)
    π = probability_vector(D)
    P = probability_matrix(K)
    πout = similar(π, size(P, 1))
    mul!(πout, P, π)
    return Categorical(πout)
end

marginalize(D::AbstractDistribution, ::IdentityKernel) = D
marginalize(D::AbstractDirac, ::IdentityKernel) = D
