ispsdparametrization(::Real) = IsPSDParametrization()
psdparametrization(::Type{T}, x::Number) where {T} = convert(real(T), real(x))
psdparametrization(x::Number) = psdparametrization(eltype(x), x)

"""
    rsqrt(x::Real)

Computes the right-square root of x.
"""
rsqrt(x::Real) = sqrt(x)

"""
    lsqrt(x::Real)

Computes the right-square root of x.
"""
lsqrt(x::Real) = rsqrt(x)

"""
    stein(Σ::Real, Φ::Number)

Computes the output of the stein  operator

    Σ ↦ Φ * Σ * Φ'.
"""
stein(Σ::Real, Φ::Number) = abs2(Φ) * Σ

"""
    stein(Σ::Real, Φ::Number, Q::Real)

Computes the output of the stein  operator

    Σ ↦ Φ * Σ * Φ' + Q.
"""
stein(Σ::Real, Φ::Number, Q::Real) = stein(Σ, Φ) + Q

"""
    schur_reduce(Π::Real, C::Number)

Returns the tuple (S, K, Σ) associated with the following (block) Schur reduction:

    [C*Π*C' C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*(C*Π*C')*[I; K]'

In terms of Kalman filtering, Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.
"""
function schur_reduce(Π::Real, C::Number)
    # this probably breaks if iszero(C) returns true
    S = abs2(C) * Π
    K = Π * adjoint(C) / S
    L = (I - K * C)
    Σ = abs2(L) * Π
    return S, K, Σ
end

"""
    schur_reduce(Π::Real, C::Number, R::Real)

Returns the tuple (S, K, Σ) associated with the following (block) Schur reduction:

    [C*Π*C'+R C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*(C*Π*C' + R)*[I; K]'

In terms of Kalman filtering, Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.
"""
function schur_reduce(Π::Real, C::Number, R::Real)
    # this probably breaks if iszero(C) && iszero(R) returns true
    S = abs2(C) * Π + R
    K = Π * adjoint(C) / S
    L = (I - K * C)
    Σ = abs2(L) * Π + abs2(K) * R
    return S, K, Σ
end
