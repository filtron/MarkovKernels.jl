# add docstr
function Normal(μ::Number, Σ::Number)
    T = promote_type(typeof(μ), typeof(Σ))
    return Normal{T}(convert(T, μ), convert(T, Σ))
end

const UvNormal{T} = Normal{T,T,T} where {T<:Number}
#const UvNormal{T,V} = Union{Normal{T,T,T},Normal{T,T,V}} where {T<:Real,V<:Complex{T}}

cov(N::UvNormal) = real(N.Σ)
var(N::UvNormal) = cov(N)

Normal{T}(N::UvNormal) where {T} = Normal(convert(T, mean(N)), convert(T, covp(N)))

rand(rng::AbstractRNG, N::UvNormal) = mean(N) + lsqrt(covp(N)) * randn(rng, eltype(N))
