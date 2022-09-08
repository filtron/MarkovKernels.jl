
# fix logdet for Hermitian matrices
function LinearAlgebra.logdet(H::Hermitian)
    mag, sign = logabsdet(H)
    resign = real(sign)
    return mag + log(resign)
end

# matrix square-roots
lsqrt(m::AbstractMatrix) = cholesky(m).L
lsqrt(m::UniformScaling) = sqrt(m)

# stein operator
stein(Σ, Φ) = Matrix(Hermitian(Φ * Σ * Φ'))
stein(Σ::T, Φ::T) where {T<:UniformScaling} = Φ * Σ * Φ'

stein(Σ, Φ, Q) = Matrix(Hermitian(Φ * Σ * Φ' + Q))
stein(Σ::T, Φ::T, Q::T) where {T<:UniformScaling} = Φ * Σ * Φ' + Q

stein(Σ, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ, A::AbstractAffineMap, Q) = stein(Σ, slope(A), Q)

# schur reduction
function schur_red(Π, C, R)
    K = Π * C'
    S = Hermitian(C * K + R)

    K = K / S

    # Joseph form
    L = (I - K * C)

    Σ = Hermitian(L * Π * L' + K * R * K')

    return Matrix(S), K, Matrix(Σ)
end

schur_red(Π, C) = schur_red(Π, C, 0.0 * I) # might be done smarter?
