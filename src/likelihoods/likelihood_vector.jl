"""
    LikelihoodVector

Type for representing a Likelihood function over categories.
"""
struct LikelihoodVector{A} <: AbstractLikelihood
    ls::A
end

"""
    LikelihoodVector(h::Likelihood{<:StochasticMatrix})

Computes a categorical likelihood from h.
"""
function LikelihoodVector(h::Likelihood{<:StochasticMatrix})
    K, y = measurement_model(h), measurement(h)
    P = probability_matrix(K)
    ls = P[y, :]
    return LikelihoodVector{typeof(ls)}(ls)
end

"""
    likelihood_vector(h::LikelihoodVector)

Computes the vector of likelihood evaluations.
"""
likelihood_vector(h::LikelihoodVector) = h.ls

function likelihood_vector(h::Likelihood{<:StochasticMatrix})
    K, y = measurement_model(h), measurement(h)
    P = probability_matrix(K)
    ls = P[y, :]
    return ls
end

log(h::LikelihoodVector, x) = log(likelihood_vector(h)[x])

similar(h::LikelihoodVector) = similar(h, eltype(likelihood_vector(h)))
similar(h::LikelihoodVector, ::Type{T}) where {T<:Real} =
    similar(h, T, axes(likelihood_vector(h)))
similar(h::LikelihoodVector, d) = similar(h, eltype(likelihood_vector(h)), d)

function similar(h::LikelihoodVector, ::Type{T}, d) where {T<:Real}
    ls = likelihood_vector(h)
    lsout = similar(ls, T, d)
    hout = LikelihoodVector(lsout)
    return hout
end

function similar(h::Likelihood{<:StochasticMatrix})
    k = measurement_model(h)
    P = probability_matrix(k)
    T = eltype(P)
    return similar(h, T)
end

function similar(h::Likelihood{<:StochasticMatrix}, ::Type{T}) where {T<:Real}
    k = measurement_model(h)
    P = probability_matrix(k)
    d = axes(P, 1)
    return similar(h, T, d)
end

function similar(h::Likelihood{<:StochasticMatrix}, d)
    k = measurement_model(h)
    P = probability_matrix(k)
    T = eltype(P)
    return similar(h, T, d)
end

function similar(h::Likelihood{<:StochasticMatrix}, ::Type{T}, d) where {T<:Real}
    k = measurement_model(h)
    P = probability_matrix(k)
    ls = similar(P, T, d)
    hout = LikelihoodVector(ls)
    return hout
end
