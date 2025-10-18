
psdcheck(::Cholesky) = IsPSD()

convert_psd_eltype(::Type{T}, C::Cholesky) where {T} = convert(Factorization{T}, C)

rsqrt(C::Cholesky) = C.uplo == 'U' ? C.U : adjoint(C.L)

function psdsimilar(C::Cholesky, ::Type{T}, d) where {T}
    factors = similar(C.factors, T, d, d)
    return Cholesky(UpperTriangular(factors))
end

function stein(
    Σ::Cholesky,
    Φ::AbstractMatrix,
    work_arr::AbstractMatrix = similar(Φ, reverse(size(Φ))),
)
    m, n = size(Φ)
    work_arr = view(work_arr, 1:n, 1:m)
    Π = psdsimilar(Σ, m)
    return stein!(Π, Σ, Φ, work_arr)
end

function stein!(
    Π::Cholesky,
    Σ::Cholesky,
    Φ::AbstractMatrix,
    work_arr::AbstractMatrix = similar(Φ, reverse(size(Φ))),
)
    m, n = size(Φ)
    work_arr = view(work_arr, 1:n, 1:m)

    mul!(work_arr, rsqrt(Σ), adjoint(Φ))
    U = positive_qrwoq!(work_arr)
    copy!(rsqrt(Π), UpperTriangular(U))
    return Π
end

# can not be made in-place because numbers are not mutable. 
function stein(
    Σ::Cholesky,
    Φ::Adjoint{<:Number,<:AbstractVector},
    work_arr::AbstractMatrix = similar(Φ, reverse(size(Φ))),
)
    m, n = size(Φ)
    work_arr = view(work_arr, 1:n, 1:m)

    mul!(work_arr, rsqrt(Σ), adjoint(Φ))
    U = positive_qrwoq!(work_arr)
    Π = abs2(U[1, 1])
    return Π
end

function stein(
    Σ::Cholesky,
    Φ::AbstractMatrix,
    Q,
    work_arr::AbstractMatrix = similar(Φ, sum(size(Φ)), size(Φ, 1)),
)
    m, n = size(Φ)
    work_arr = view(work_arr, 1:(n+m), 1:m)
    Π = psdsimilar(Σ, m)
    return stein!(Π, Σ, Φ, Q, work_arr)
end

function stein!(
    Π::Cholesky,
    Σ::Cholesky,
    Φ::AbstractMatrix,
    Q,
    work_arr::AbstractMatrix = similar(Φ, sum(size(Φ)), size(Φ, 1)),
)
    m, n = size(Φ)
    work_arr = view(work_arr, 1:(n+m), 1:m)

    mul!(view(work_arr, 1:n, 1:m), rsqrt(Σ), adjoint(Φ))
    copyto!(view(work_arr, (n+1):(n+m), 1:m), rsqrt(Q))

    U = positive_qrwoq!(work_arr)
    copy!(rsqrt(Π), UpperTriangular(U))
    return Π
end

# can not be made in-place because numbers are not mutable. 
function stein(Σ::Cholesky, Φ::Adjoint{<:Number,<:AbstractVector}, Q::Number)
    m, n = size(Φ)
    work_arr = similar(Φ, n + m, m)

    mul!(view(work_arr, 1:n, 1:m), rsqrt(Σ), adjoint(Φ))
    view(work_arr, (n+1):(n+m), 1:m) .= rsqrt(Q)

    U = positive_qrwoq!(work_arr)
    Π = abs2(U[1, 1])
    return Π
end

# can not be made in-place because numbers are not mutable. 
function stein(Σ::Cholesky, Φ::Adjoint{<:Number,<:AbstractVector}, Q::UniformScaling)
    return stein(Σ, Φ, Q.λ)
end

function _schur_reduce(Π::Cholesky, C::AbstractMatrix)
    m, n = size(C)
    work_arr = similar(C, n + m, n + m)

    mul!(view(work_arr, 1:n, 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, 1:n, (m+1):(n+m)) .= rsqrt(Π)
    view(work_arr, (n+1):(n+m), 1:(n+m)) .= zero(eltype(work_arr))
    positive_qrwoq!(view(work_arr, 1:n, 1:(n+m)))

    yidx, xidx = 1:m, (m+1):(n+m)
    S = @inbounds Cholesky(UpperTriangular(work_arr[yidx, yidx]))
    Σ = @inbounds Cholesky(UpperTriangular(work_arr[xidx, xidx]))

    Kadj = @inbounds view(work_arr, yidx, xidx)
    K = @inbounds view(work_arr, xidx, yidx)
    K .= adjoint(Kadj)
    K = copy(K) # copy because we dont want to return SubArray

    return S, K, Σ
end

function _schur_reduce(Π::Cholesky, C::Adjoint{<:Number,<:AbstractVector})
    m, n = size(C)
    work_arr = similar(C, n + m, n + m)

    mul!(view(work_arr, 1:n, 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, 1:n, (m+1):(n+m)) .= rsqrt(Π)
    view(work_arr, (n+1):(n+m), 1:(n+m)) .= zero(eltype(work_arr))
    positive_qrwoq!(view(work_arr, 1:n, 1:(n+m)))

    yidx, xidx = 1, (m+1):(n+m) # yidx = 1:m = 1:1 but set to 1 so relevant SubArrays become vectors

    Ssqrt = @inbounds work_arr[1, 1]
    Σ = @inbounds Cholesky(UpperTriangular(work_arr[xidx, xidx]))

    K = @inbounds conj.(view(work_arr, yidx, xidx))
    S = abs2(Ssqrt)
    K = copy(K)

    return S, K, Σ
end

function _schur_reduce(Π::Cholesky, C::AbstractMatrix, R)
    m, n = size(C)
    work_arr = similar(C, n + m, n + m)

    #view(work_arr, 1:m, 1:m) .= rsqrt(R)
    copyto!(view(work_arr, 1:m, 1:m), rsqrt(R))
    view(work_arr, 1:m, (m+1):(n+m)) .= zero(eltype(work_arr))
    mul!(view(work_arr, (m+1):(n+m), 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, (m+1):(n+m), (m+1):(n+m)) .= rsqrt(Π)
    positive_qrwoq!(work_arr)

    yidx, xidx = 1:m, (m+1):(n+m)
    S = @inbounds Cholesky(UpperTriangular(work_arr[yidx, yidx]))
    Σ = @inbounds Cholesky(UpperTriangular(work_arr[xidx, xidx]))

    Kadj = @inbounds view(work_arr, yidx, xidx)
    K = @inbounds view(work_arr, xidx, yidx)
    K .= adjoint(Kadj)
    K = copy(K)

    return S, K, Σ
end

function _schur_reduce(Π::Cholesky, C::Adjoint{<:Number,<:AbstractVector}, R::Number)
    m, n = size(C) # m = 1
    work_arr = similar(C, n + m, n + m)

    view(work_arr, 1:m, 1:m) .= rsqrt(R)
    view(work_arr, 1:m, (m+1):(n+m)) .= zero(eltype(work_arr))
    mul!(view(work_arr, (m+1):(n+m), 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, (m+1):(n+m), (m+1):(n+m)) .= rsqrt(Π)
    positive_qrwoq!(work_arr)

    yidx, xidx = 1, (m+1):(n+m) # yidx = 1:m = 1:1 but set to 1 so relevant SubArrays become vectors

    Ssqrt = @inbounds work_arr[1, 1]
    Σ = @inbounds Cholesky(UpperTriangular(work_arr[xidx, xidx]))

    K = @inbounds conj.(view(work_arr, yidx, xidx))
    S = abs2(Ssqrt)
    K = copy(K)

    return S, K, Σ
end

function _schur_reduce(
    Π::Cholesky,
    C::Adjoint{<:Number,<:AbstractVector},
    R::UniformScaling,
)
    return _schur_reduce(Π, C, R.λ)
end

function schur_reduce(Π::Cholesky, C::AbstractMatrix)
    S, K, Σ = _schur_reduce(Π, C)
    K = rdiv!(K, lsqrt(S))
    return S, K, Σ
end

function schur_reduce(Π::Cholesky, C::AbstractMatrix, R)
    S, K, Σ = _schur_reduce(Π, C, R)
    K = rdiv!(K, lsqrt(S))
    return S, K, Σ
end
