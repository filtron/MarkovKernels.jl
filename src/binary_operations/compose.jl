"""
    compose(K2::AbstractMarkovKernel, K1::AbstractMarkovKernel)

Computes K3, the composition of K2 ∘ K1 i.e.,

K3(y,x) = ∫ K2(y,z) K1(z,x) dz.

See also [`∘`](@ref)
"""
function compose(::AbstractMarkovKernel, ::AbstractMarkovKernel) end

compose(K2::AffineHomoskedasticNormalKernel, K1::AffineHomoskedasticNormalKernel) =
    NormalKernel(compose(mean(K2), mean(K1)), stein(covp(K1), mean(K2), covp(K2)))

compose(K2::AffineHomoskedasticNormalKernel, K1::AffineDiracKernel) =
    NormalKernel(compose(mean(K2), mean(K1)), covp(K2))

compose(K2::AffineHeteroskedasticNormalKernel, K1::AffineDiracKernel) =
    NormalKernel(compose(mean(K2), mean(K1)), covp(K2) ∘ mean(K1))

compose(K2::AffineDiracKernel, K1::AffineDiracKernel) =
    DiracKernel(compose(mean(K2), mean(K1)))

compose(K2::AffineDiracKernel, K1::AffineHomoskedasticNormalKernel) =
    NormalKernel(compose(mean(K2), mean(K1)), stein(covp(K1), mean(K2)))

function compose(K2::StochasticMatrix, K1::StochasticMatrix)
    P2 = probability_matrix(K2)
    P1 = probability_matrix(K1)
    m, n = size(P2, 1), size(P1, 2)
    P3 = similar(P2, m, n)
    mul!(P3, P2, P1)
    return StochasticMatrix(P3)
end

compose(K2::AbstractMarkovKernel, ::IdentityKernel) = K2
compose(::IdentityKernel, K1::AbstractMarkovKernel) = K1
compose(K2::IdentityKernel, ::IdentityKernel) = K2

"""
    ∘(K2::AbstractMarkovKernel, K1::AbstractMarkovKernel)

Computes K3, the composition of K2 ∘ K1 i.e.,

    K3(y,x) = ∫ K2(y,z) K1(z,x) dz.

See also [`compose`](@ref)
"""
∘(K2::AbstractMarkovKernel, K1::AbstractMarkovKernel) = compose(K2, K1)

"""
    compose(L2::AbstractLikelihood, L1::AbstractLiklelihood)

Computes L3, the composition of L2 ∘ L1 i.e.,

L3(x) = L1(x) * L2(x)

See also [`∘`](@ref)
"""
function compose(::AbstractLikelihood, ::AbstractLikelihood) end

compose(L1::AbstractLikelihood, L2::FlatLikelihood) = L1
compose(L1::FlatLikelihood, L2::AbstractLikelihood) = compose(L2, L1)
compose(::FlatLikelihood, ::FlatLikelihood) = FlatLikelihood()

function compose(L1::CategoricalLikelihood, L2::CategoricalLikelihood)
    l1 = likelihood_vector(L1)
    l2 = likelihood_vector(L2)
    l3 = similar(l1)
    l3 .= l1 .* l2
    return CategoricalLikelihood(l3)
end

compose(L1::CategoricalLikelihood, L2::Likelihood{<:StochasticMatrix}) =
    compose(L1, CategoricalLikelihood(L2))
compose(L1::Likelihood{<:StochasticMatrix}, L2::CategoricalLikelihood) = compose(L2, L1)
compose(L1::Likelihood{<:StochasticMatrix}, L2::Likelihood{<:StochasticMatrix}) =
    compose(CategoricalLikelihood(L1), CategoricalLikelihood(L2))

function compose(L1::LogQuadraticLikelihood, L2::LogQuadraticLikelihood)
    logc1, y1, C1 = L1
    m1, n1 = size(C1)

    logc2, y2, C2 = L2
    m2, n2 = size(C2)

    Chat = vcat(C1, C2)
    yhat = vcat(y1, y2)
    T = eltype(yhat)

    logcbar = logc1 + logc2
    F = qr!(Chat)
    Cbar = F.R
    y3 = adjoint(F.Q) * yhat

    ybar = y3[1:min(m1+m2, n1)]
    e_norm_sqr = norm(y3)^2 - norm(ybar)^2
    logcbar = logcbar - _nscale(T) * e_norm_sqr

    return LogQuadraticLikelihood(logcbar, ybar, Cbar)
end

compose(L1::LogQuadraticLikelihood, L2::Likelihood{<:AffineHomoskedasticNormalKernel}) =
    compose(L1, LogQuadraticLikelihood(L2))
compose(L1::Likelihood{<:AffineHomoskedasticNormalKernel}, L2::LogQuadraticLikelihood) =
    compose(L2, L1)
compose(
    L1::Likelihood{<:AffineHomoskedasticNormalKernel},
    L2::Likelihood{<:AffineHomoskedasticNormalKernel},
) = compose(LogQuadraticLikelihood(L1), LogQuadraticLikelihood(L2))

"""
    ∘(K2::AbstractLikelihood, K1::AbstractLikelihood)

Computes L3, the composition of L2 ∘ L1 i.e.,

L3(x) = L1(x) * L2(x)

See also [`compose`](@ref)
"""
∘(L1::AbstractLikelihood, L2::AbstractLikelihood) = compose(L1, L2)
