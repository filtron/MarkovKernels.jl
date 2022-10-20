"""
invert(N::AbstractNorma, K::AffineNormalKernel)

Returns the inverted factorisation of the joint distirbution P(y,x) = N(x)*K(y, x) i.e

P(y,x) = Nout(y)*Kout(x,y)
"""
function invert(N::AbstractNormal{T}, K::AffineNormalKernel{T}) where {T}
    pred = mean(K)(mean(N))
    S, G, Σ = schur_red(covp(N), mean(K), covp(K))

    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineCorrector(G, mean(N), pred), Σ)

    return Nout, Kout
end

function invert(N::AbstractNormal{T}, K::AffineDiracKernel{T}) where {T}
    pred = mean(K)(mean(N))
    S, G, Σ = schur_red(covp(N), mean(K))

    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineCorrector(G, mean(N), pred), Σ)

    return Nout, Kout
end
