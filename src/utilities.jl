
# fix logdet for Hermitian matrices
function logdet(m::Hermitian)
    lad = logabsdet(m)
    real(lad[2]) >= 0 ? lad[1] : -Inf
end

# matrix square-roots
lsqrt(m::AbstractMatrix) = cholesky(m).L
rsqrt(m) = cholesky(Hermitian(m)).U

# trace of ratio
trdiv(Σ1, Σ2) = tr(Σ2 \ Σ1) #norm_sqr( lsqrt(Σ2) \ lsqrt(Σ1) )

# stein operator

stein(Σ::AbstractMatrix, Φ::AbstractMatrix) = Matrix(Hermitian(Φ * Σ * Φ'))
stein(Σ::AbstractMatrix, Φ::AbstractMatrix, Q::AbstractMatrix) =
    Matrix(Hermitian(Φ * Σ * Φ' + Q))
stein(Σ::AbstractMatrix, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ::AbstractMatrix, A::AbstractAffineMap, Q::AbstractMatrix) = stein(Σ, slope(A), Q)

# schur reduction
function schur_red(Π::AbstractMatrix, C, R)
    K = Π * C'
    S = Hermitian(C * K + R)

    K = K / S

    # Joseph form
    L = (I - K * C)

    Σ = Hermitian(L * Π * L' + K * R * K')

    return Matrix(S), K, Matrix(Σ)
end

schur_red(Π::AbstractMatrix, C) = schur_red(Π, C, 0.0 * I) # might be done smarter?
