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
    Kout = NormalKernel(AffineMap(G, mean(N) - G * pred), Σ)
    return Nout, Kout
end

function invert(N::AbstractNormal, K::AffineDiracKernel)
    pred = mean(K)(mean(N))
    S, G, Σ = schur_reduce(covparam(N), mean(K))
    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineMap(G, mean(N) - G * pred), Σ)
    return Nout, Kout
end

#=
function invert(d::ProbabilityVector, k::AbstractStochasticMatrix)
    π = probability_vector(d)
    P = probability_matrix(k)

    πout = similar(π, size(P, 1))
    πout = mul!(πout, P, π)
    dout = ProbabilityVector(πout)

    Pout = similar(adjoint(P))
    for i in axes(Pout, 1), j in axes(Pout, 2)
        Pout[i, j] = P[j, i] * π[i] / πout[j]
    end
    kout = StochasticMatrix(Pout)

    return dout, kout
end
=#

function invert(d::ProbabilityVector, k::AbstractStochasticMatrix)
    P = probability_matrix(k)
    dout = similar(d, size(k, 1))
    kout = similar(k, reverse(size(k)))
    return invert!(dout, kout, d, k)
end

# we can destroy k here 
function invert!(
    dout::ProbabilityVector,
    kout::AbstractStochasticMatrix,
    d::ProbabilityVector,
    k::AbstractStochasticMatrix,
)
    dout = forward_operator!(dout, k, d)

    π = probability_vector(d)
    πout = probability_vector(dout)
    P = probability_matrix(k)
    Pout = probability_matrix(kout)
    for i in axes(Pout, 1), j in axes(Pout, 2)
        Pout[i, j] = P[j, i] * π[i] / πout[j]
    end
    return dout, kout
end

invert(D::AbstractDistribution, K::IdentityKernel) = D, K
