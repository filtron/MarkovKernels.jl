
psdcheck(::Cholesky) = IsPSD()

convert_psd_eltype(::Type{T}, C::Cholesky) where {T} = convert(Factorization{T}, C)

rsqrt(C::Cholesky) = C.uplo == 'U' ? C.U : adjoint(C.L)

function stein(Σ::Cholesky, Φ::AbstractMatrix)
    m, n = size(Φ)
    work_arr = similar(Φ, n, m)

    mul!(work_arr, rsqrt(Σ), adjoint(Φ))
    U = positive_qrwoq!(work_arr)
    Π = Cholesky(UpperTriangular(copy(U)))
    return Π
end

function stein(Σ::Cholesky, Φ::Adjoint{<:Number,<:AbstractVector})
    m, n = size(Φ)
    work_arr = similar(Φ, n, m)

    mul!(work_arr, rsqrt(Σ), adjoint(Φ))
    U = positive_qrwoq!(work_arr)
    Π = abs2(U[1, 1])
    return Π
end

function stein(Σ::Cholesky, Φ::AbstractMatrix, Q)
    m, n = size(Φ)
    work_arr = similar(Φ, n + m, m)

    mul!(view(work_arr, 1:n, 1:m), rsqrt(Σ), adjoint(Φ))
    view(work_arr, n+1:n+m, 1:m) .= rsqrt(Q)

    U = positive_qrwoq!(work_arr)
    Π = Cholesky(UpperTriangular(copy(U)))
    return Π
end

function stein(Σ::Cholesky, Φ::Adjoint{<:Number,<:AbstractVector}, Q::Number)
    m, n = size(Φ)
    work_arr = similar(Φ, n + m, m)

    mul!(view(work_arr, 1:n, 1:m), rsqrt(Σ), adjoint(Φ))
    view(work_arr, n+1:n+m, 1:m) .= rsqrt(Q)

    U = positive_qrwoq!(work_arr)
    Π = abs2(U[1, 1])
    return Π
end

function schur_reduce(Π::Cholesky, C::AbstractMatrix)
    m, n = size(C)
    work_arr = similar(C, n + m, n + m)

    mul!(view(work_arr, 1:n, 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, 1:n, m+1:n+m) .= rsqrt(Π)
    view(work_arr, n+1:n+m, 1:n+m) .= zero(eltype(work_arr))
    positive_qrwoq!(view(work_arr, 1:n, 1:n+m))

    yidx, xidx = 1:m, m+1:n+m
    S = @inbounds Cholesky(UpperTriangular(work_arr[yidx, yidx]))
    Σ = @inbounds Cholesky(UpperTriangular(work_arr[xidx, xidx]))

    Kadj = @inbounds view(work_arr, yidx, xidx)
    K = @inbounds view(work_arr, xidx, yidx)
    K .= adjoint(Kadj)
    K = rdiv!(K, lsqrt(S))
    K = copy(K) # copy because we dont want to return SubArray

    return S, K, Σ
end

function schur_reduce(Π::Cholesky, C::Adjoint{<:Number,<:AbstractVector})
    m, n = size(C)
    work_arr = similar(C, n + m, n + m)

    mul!(view(work_arr, 1:n, 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, 1:n, m+1:n+m) .= rsqrt(Π)
    view(work_arr, n+1:n+m, 1:n+m) .= zero(eltype(work_arr))
    positive_qrwoq!(view(work_arr, 1:n, 1:n+m))

    yidx, xidx = 1, m+1:n+m # yidx = 1:m = 1:1 but set to 1 so relevant SubArrays become vectors

    Ssqrt = @inbounds work_arr[1, 1]
    Σ = @inbounds Cholesky(UpperTriangular(work_arr[xidx, xidx]))

    K = @inbounds conj.(view(work_arr, yidx, xidx))
    K = rdiv!(K, Ssqrt)
    S = abs2(Ssqrt)
    K = copy(K) # copy because we dont want to return SubArray

    return S, K, Σ
end

function schur_reduce(Π::Cholesky, C::AbstractMatrix, R)
    m, n = size(C)
    work_arr = similar(C, n + m, n + m)

    view(work_arr, 1:m, 1:m) .= rsqrt(R)
    view(work_arr, 1:m, m+1:n+m) .= zero(eltype(work_arr))
    mul!(view(work_arr, m+1:n+m, 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, m+1:n+m, m+1:n+m) .= rsqrt(Π)
    positive_qrwoq!(work_arr)

    yidx, xidx = 1:m, m+1:n+m
    S = @inbounds Cholesky(UpperTriangular(work_arr[yidx, yidx]))
    Σ = @inbounds Cholesky(UpperTriangular(work_arr[xidx, xidx]))

    Kadj = @inbounds view(work_arr, yidx, xidx)
    K = @inbounds view(work_arr, xidx, yidx)
    K .= adjoint(Kadj)
    K = rdiv!(K, lsqrt(S))
    K = copy(K) # copy because we dont want to return SubArray

    return S, K, Σ
end

function schur_reduce(Π::Cholesky, C::Adjoint{<:Number,<:AbstractVector}, R::Number)
    m, n = size(C) # m = 1
    work_arr = similar(C, n + m, n + m)

    view(work_arr, 1:m, 1:m) .= rsqrt(R)
    view(work_arr, 1:m, m+1:n+m) .= zero(eltype(work_arr))
    mul!(view(work_arr, m+1:n+m, 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, m+1:n+m, m+1:n+m) .= rsqrt(Π)
    positive_qrwoq!(work_arr)

    yidx, xidx = 1, m+1:n+m # yidx = 1:m = 1:1 but set to 1 so relevant SubArrays become vectors

    Ssqrt = @inbounds work_arr[1, 1]
    Σ = @inbounds Cholesky(UpperTriangular(work_arr[xidx, xidx]))

    K = @inbounds conj.(view(work_arr, yidx, xidx))
    K = rdiv!(K, Ssqrt)
    S = abs2(Ssqrt)
    K = copy(K) # copy because we dont want to return SubArray

    return S, K, Σ
end
