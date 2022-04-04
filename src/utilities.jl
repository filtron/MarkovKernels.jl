

# fix logdet for Hermitian matrices
logdet(m::Hermitian) = logabsdet(m)[1]

# matrix square-roots
lsqrt(m::AbstractMatrix) = cholesky(Hermitian(m)).L
rsqrt(m) = cholesky(Hermitian(m)).U

# trace of ratio
trdiv(Σ1,Σ2) = tr(Σ2 \ Σ1) #norm_sqr( lsqrt(Σ2) \ lsqrt(Σ1) )