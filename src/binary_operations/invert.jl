"""
invert(D::AbstractDistribution, K::AbstractMarkovKernel)

Computes D2, K2, such that D(x)K(y, x) = D2(y)K2(x, y), i.e., an inverted factorisation of D, K.
"""
function invert(N::AbstractNormal, K::AffineHomoskedasticNormalKernel)
    pred = mean(K)(mean(N))
    S, G, Σ = schur_reduce(covp(N), mean(K), covp(K))
    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineCorrector(G, mean(N), pred), Σ)
    return Nout, Kout
end

function invert(N::AbstractNormal, K::AffineDiracKernel)
    pred = mean(K)(mean(N))
    S, G, Σ = schur_reduce(covp(N), mean(K))
    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineCorrector(G, mean(N), pred), Σ)
    return Nout, Kout
end

function invert(C::Categorical, K::AbstractStochasticMatrix)
    π = probability_vector(C)
    P = probability_matrix(K)

    πout = similar(π)
    πout = mul!(πout, P, π)
    Cout = Categorical(πout)

    Pout = similar(adjoint(P))
    for i in axes(Pout, 1), j in axes(Pout, 2)
        Pout[i, j] = P[j, i] * π[i] / πout[j]
    end
    Kout = StochasticMatrix(Pout)

    return Cout, Kout
end

invert(D::AbstractDistribution, K::IdentityKernel) = D, K
