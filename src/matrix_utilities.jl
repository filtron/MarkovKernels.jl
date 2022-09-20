
# fix logdet for Hermitian matrices
function LinearAlgebra.logdet(H::Hermitian)
    mag, sign = logabsdet(H)
    resign = real(sign)
    return mag + log(resign)
end

# matrix square-roots
lsqrt(m::AbstractMatrix) = cholesky(m).L
lsqrt(m::UniformScaling) = sqrt(m)

# project matrix onto symmetric matrix
symmetrise(Σ) = Σ
symmetrise(Σ::Matrix{T}) where T = T <: Real ? Symmetric(Σ) : Hermitian(Σ)

# stein operator
stein(Σ, Φ) = symmetrise(Φ * Σ * Φ')
stein(Σ, Φ, Q) = symmetrise(Φ * Σ * Φ' + Q)

stein(Σ, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ, A::AbstractAffineMap, Q) = stein(Σ, slope(A), Q)

# schur reduction
function schur_red(Π, C, R)
    K = Π * C'
    S = symmetrise(C * K + R)

    K = K / S

    # Joseph form
    L = (I - K * C)

    Σ = symmetrise(L * Π * L' + K * R * K')

    return S, K, Σ
end

schur_red(Π, C) = schur_red(Π, C, 0.0 * I) # might be done smarter?
