"""
    psdcheck(A)

Returns IsPSD() if A is a PSDParametrization otherwise IsNotPSD()
"""
psdcheck(::Any) = IsNotPSD()

"""
    convert_psd_eltype(::Type{T}, P)

Wraps P in a psd paramtrization of eltype T.
If P is already a type of psd paramtrization, then just the eltype is converted.
"""
function convert_psd_eltype(::Type{T}, P) where {T} end

convert_psd_eltype(P) = convert_psd_eltype(eltype(P), P)

"""
    rsqrt(Σ)

Computes a right square-root of Σ.
"""
function rsqrt(Σ) end

"""
    lsqrt(Σ)

Computes a left square-root of Σ.
"""
lsqrt(Σ) = adjoint(rsqrt(Σ))

"""
    stein(Σ, Φ, [Q])

Computes the output of the stein  operator

    Σ ↦ Φ * Σ * Φ' + Q.
"""
function stein(Σ, Φ, Q) end
function stein(Σ, Φ) end

"""
    schur_reduce(Π, C, [R])

Computes the tuple (S, K, Σ) associated with the following (block) Schur reduction:

    [C*Π*C'+R C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*(C*Π*C' + R)*[I; K]'

In terms of Kalman filtering, Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.
"""
function schur_reduce(Π, C, R) end
function schur_reduce(Π, C) end
