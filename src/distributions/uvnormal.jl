# add docstr
function Normal(μ::Number, Σ::Number)
    T = promote_type(typeof(μ), typeof(Σ))
    return Normal{T}(convert(T, μ), convert(T, Σ))
end

const UvNormal{T} = Normal{T,T,T} where {T<:Number}

cov(N::UvNormal) = N.Σ
var(N::UvNormal) = N.Σ

Normal{T}(N::UvNormal) where {T} = Normal(convert(T, mean(N)), convert(T, cov(N)))

rand(rng::AbstractRNG, N::UvNormal) =
    mean(N) + lsqrt(covp(N)) * randn(rng, eltype(N))