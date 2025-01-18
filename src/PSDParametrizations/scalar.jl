psdcheck(::Real) = IsPSD()
convert_psd_eltype(::Type{T}, x::Number) where {T} = convert(real(T), real(x))
convert_psd_eltype(x::Number) = convert_psd_eltype(eltype(x), x)

rsqrt(x::Real) = sqrt(x)
lsqrt(x::Real) = rsqrt(x)

stein(Σ::Real, Φ::Number) = abs2(Φ) * Σ
stein(Σ::Real, Φ::Number, Q::Real) = stein(Σ, Φ) + Q
stein(Σ::Real, Φ::Number, Q::UniformScaling) = stein(Σ, Φ, Q.λ)

function schur_reduce(Π::Real, C::Number)
    # this probably breaks if iszero(C) returns true
    S = abs2(C) * Π
    K = adjoint(C) / abs2(C)
    Σ = zero(Π)
    return S, K, Σ
end

function schur_reduce(Π::Real, C::Number, R::Real)
    # this probably breaks if iszero(C) && iszero(R) returns true
    S = abs2(C) * Π + R
    K = Π * adjoint(C) / S
    L = (I - K * C)
    Σ = abs2(L) * Π + abs2(K) * R
    return S, K, Σ
end

schur_reduce(Π::Real, C::Number, R::UniformScaling) = schur_reduce(Π, C, R.λ)
