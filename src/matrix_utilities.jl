
# fix logdet for Hermitian matrices
function LinearAlgebra.logdet(H::Hermitian)
    mag, sign = logabsdet(H)
    resign = real(sign)
    return mag + log(resign)
end

"""
    qr2chol(pre_array::AbstractMatrix)

Computes the upper triangular cholesky factor of the matrix pre_array'*pre_array
"""
function qr2chol(pre_array)
    right = qr(pre_array).R
    right_pos = conj.( sign.( Diagonal(right) ) )*right
    return  UpperTriangular(right_pos)
end

# matrix square-roots
lsqrt(m::AbstractMatrix) = cholesky(m).L
lsqrt(m::UniformScaling) = sqrt(m)
lsqrt(C::Cholesky) = C.L

# project matrix onto symmetric matrix
symmetrise(Σ::AbstractMatrix{T}) where {T} = T <: Real ? Symmetric(Σ) : Hermitian(Σ)
symmetrise(Σ::UniformScaling) = Σ
symmetrise(Σ::Diagonal) = Σ
symmetrise(C::Cholesky) = C

"""
    stein()
"""
stein(Σ, Φ) = symmetrise(Φ * Σ * Φ')
stein(Σ::Cholesky, Φ) = Cholesky( qr2chol(Σ.U*Φ') )

stein(Σ, Φ, Q) = symmetrise(Φ * Σ * Φ' + Q)
stein(Σ::Cholesky, Φ, Q) = Cholesky( qr2chol([Σ.U*Φ'; lsqrt(Q)']) ) # add UniformScaling / Cholesky, Diagonal / Cholesky ?

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


function schur_red(Π::Cholesky, C, R::Cholesky)

    # some hacking required to support C without size ...
    ny, nx  = size(C)

    pre_array = [R.U zeros(ny, nx); Π.U*C' Π.U]
    post_array = qr2chol(pre_array)

    S = Cholesky(UpperTriangular(post_array[1:ny, 1:ny]))
    Kt = post_array[1:ny, ny + 1:ny + nx] # should divide by square-root of S or so here...
    Σ = Cholesky(UpperTriangular(post_array[ny + 1:nx, ny + 1:ny + nx]))

    return S, Kt', Σ
end

