
rsqrt(x::Real) = sqrt(x)
lsqrt(x::Real) = rsqrt(x)

stein(Σ::Real, Φ::Number) = abs2(Φ) * Σ
stein(Σ::Real, Φ::Number, Q::Real) = stein(Σ, Φ) + Q

# this probably breaks if iszero(C) returns true 
function schur_reduce(Π::Real, C::Number)
    S = abs2(C) * Π
    K = Π * adjoint(C) / S
    L = (I - K * C)
    Σ = abs2(L) * Π
    return S, K, Σ
end

# this probably breaks if iszero(C) && iszero(R) returns true 
function schur_reduce(Π::Real, C::Number, R::Real)
    S = abs2(C) * Π + R
    K = Π * adjoint(C) / S
    L = (I - K * C)
    Σ = abs2(L) * Π + abs2(K) * R
    return S, K, Σ
end
