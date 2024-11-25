"""
    htransform(K::AbstractMarkov, L::AbstractLikelihood)

Computes a Markov kernel Kout and Likelihood Lout such that

Lout(z) = ∫ L(x) K(x, z) dx, and Kout(x, z) = L(x) K(x, z) / Lout(z)

"""
function htransform(::AbstractMarkovKernel, ::AbstractLikelihood) end

function htransform(K::StochasticMatrix, L::CategoricalLikelihood)
    ls = likelihood_vector(L)
    P = probability_matrix(K)

    lsout = similar(P, size(P, 2))
    lsout = mul!(lsout, adjoint(P), ls)

    Pout = similar(P)

    for i in axes(P, 1), j in axes(P, 2)
        Pout[i, j] = ls[i] * P[i, j] / lsout[j]
    end

    Lout = CategoricalLikelihood(lsout)
    Kout = StochasticMatrix(Pout)
    return Kout, Lout
end

function htransform(K::StochasticMatrix, L::Likelihood{<:StochasticMatrix})
    Kobs, y = measurement_model(L), measurement(L)
    ls = view(probability_matrix(Kobs), y, :)
    Ltmp = CategoricalLikelihood(ls)
    return htransform(K, Ltmp)
end
