"""
    Normal(μ::Number, Σ::Real)

Creates a univariate Normal distribution with mean μ and variance Σ. 
"""
function Normal(μ::Number, Σ::Real)
    T = promote_type(typeof(μ), typeof(Σ))
    return Normal{T}(convert(T, μ), convert(real(T), Σ))
end

const UvNormal{T,V} = Union{Normal{V,V,V},Normal{T,T,V}} where {V<:Real,T<:Complex{V}}

cov(N::UvNormal) = N.Σ
var(N::UvNormal) = cov(N)

Normal{T}(N::UvNormal) where {T} = Normal(convert(T, mean(N)), convert(real(T), covp(N)))

rand(rng::AbstractRNG, N::UvNormal) = mean(N) + lsqrt(covp(N)) * randn(rng, eltype(N))
