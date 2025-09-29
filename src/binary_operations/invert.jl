"""
invert(D::AbstractDistribution, K::AbstractMarkovKernel)

Computes a new distribution and Markov kernel such that

Dout(y) = ∫ K(y, x) D(x) dx, and Kout(x, y) = K(y, x) * D(x) / Dout(y)
"""
function invert(::AbstractDistribution, ::AbstractMarkovKernel) end

function invert(N::AbstractNormal, K::AffineHomoskedasticNormalKernel)
    pred = mean(K)(mean(N))
    S, G, Σ = schur_reduce(covparam(N), mean(K), covparam(K))
    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineCorrector(G, mean(N), pred), Σ)
    return Nout, Kout
end

function invert(N::AbstractNormal, K::AffineDiracKernel)
    pred = mean(K)(mean(N))
    S, G, Σ = schur_reduce(covparam(N), mean(K))
    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineCorrector(G, mean(N), pred), Σ)
    return Nout, Kout
end

function invert(C::Categorical, K::AbstractStochasticMatrix)
    π = probability_vector(C)
    P = probability_matrix(K)

    πout = similar(π, size(P, 1))
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
