"""
invert(D::AbstractDistribution, K::AbstractMarkovKernel)

Computes D2, K2, such that D(x)K(y, x) = D2(y)K2(x, y), i.e., an inverted factorisation of D, K.
"""
function invert(::AbstractDistribution, ::AbstractMarkovKernel) end

function invert(N::AbstractNormal, K::AffineHomoskedasticNormalKernel)
    pred = mean(K)(mean(N))
    S, G, Σ = schur_reduce(covp(N), mean(K), covp(K))
    Nout = Normal(pred, S)
    Kout = HomoskedasticNormalKernel(AffineCorrector(G, mean(N), pred), Σ)
    return Nout, Kout
end

function invert(N::AbstractNormal{T}, K::AffineDiracKernel{T}) where {T}
    pred = mean(K)(mean(N))
    S, G, Σ = schur_reduce(covp(N), mean(K))
    Nout = Normal(pred, S)
    Kout = HomoskedasticNormalKernel(AffineCorrector(G, mean(N), pred), Σ)
    return Nout, Kout
end

invert(D::AbstractDistribution, K::IdentityKernel) = D, K

# remove these
function invert(N::AbstractNormal{T}, K::AffineNormalKernel{T}) where {T}
    pred = mean(K)(mean(N))
    S, G, Σ = schur_reduce(covp(N), mean(K), covp(K))
    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineCorrector(G, mean(N), pred), Σ)
    return Nout, Kout
end

function invert(N::AbstractNormal{T}, K::AffineDiracKernel{T}) where {T}
    pred = mean(K)(mean(N))
    S, G, Σ = schur_reduce(covp(N), mean(K))
    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineCorrector(G, mean(N), pred), Σ)
    return Nout, Kout
end
#
