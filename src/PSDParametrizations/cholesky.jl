
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
function stein(
    Σ::Cholesky,
    Φ::Adjoint{<:Number,<:AbstractVector},
    Q::Number,
    work_arr::AbstractMatrix = similar(Φ, sum(size(Φ)), size(Φ, 1)),
)
    m, n = size(Φ)
    work_arr = view(work_arr, 1:(n+m), 1:m)
    #work_arr = similar(Φ, n + m, m)

    mul!(view(work_arr, 1:n, 1:m), rsqrt(Σ), adjoint(Φ))
    view(work_arr, (n+1):(n+m), 1:m) .= rsqrt(Q)

    U = positive_qrwoq!(work_arr)
    Π = abs2(U[1, 1])
    return Π
end

# can not be made in-place because numbers are not mutable. 
function stein(
    Σ::Cholesky,
    Φ::Adjoint{<:Number,<:AbstractVector},
    Q::UniformScaling,
    work_arr::AbstractMatrix = similar(Φ, sum(size(Φ)), size(Φ, 1)),
)
    return stein(Σ, Φ, Q.λ, work_arr)
end

# to be deleted (breaks htransform_and_likelihood?)
function _schur_reduce(Π::Cholesky, C::AbstractMatrix)
    m, n = size(C)
    work_arr = similar(C, n + m, n + m)

    S = psdsimilar(Π, m)
    K = similar(adjoint(C))
    Σ = psdsimilar(Π, n)
    return _schur_reduce!(S, K, Σ, Π, C, work_arr)
end

function schur_reduce(
    Π::Cholesky,
    C::AbstractMatrix,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    m, n = size(C)
    S = psdsimilar(Π, m)
    K = similar(adjoint(C))
    Σ = psdsimilar(Π, n)
    return schur_reduce!(S, K, Σ, Π, C, work_arr)
end

function schur_reduce(
    Π::Cholesky,
    C::Adjoint{<:Number,<:AbstractVector},
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    m, n = size(C)
    K = similar(adjoint(C))
    Σ = psdsimilar(Π, n)
    return schur_reduce!(K, Σ, Π, C, work_arr)
end

function schur_reduce!(
    S::Cholesky,
    K::AbstractMatrix,
    Σ::Cholesky,
    Π::Cholesky,
    C::AbstractMatrix,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    S, K, Σ = _schur_reduce!(S, K, Σ, Π, C, work_arr)
    K = rdiv!(K, lsqrt(S))
    return S, K, Σ
end

# can not pre-allocate S here because it is a number 
function schur_reduce!(
    K::AbstractVector,
    Σ::Cholesky,
    Π::Cholesky,
    C::Adjoint{<:Number,<:AbstractVector},
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    S, K, Σ = _schur_reduce!(Σ, K, Π, C, work_arr)
    K = rdiv!(K, lsqrt(S))
    return S, K, Σ
end

function _schur_reduce!(
    S::Cholesky,
    K::AbstractMatrix,
    Σ::Cholesky,
    Π::Cholesky,
    C::AbstractMatrix,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    m, n = size(C)
    work_arr = view(work_arr, 1:(n+m), 1:(n+m))

    mul!(view(work_arr, 1:n, 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, 1:n, (m+1):(n+m)) .= rsqrt(Π)
    view(work_arr, (n+1):(n+m), 1:(n+m)) .= zero(eltype(work_arr))
    positive_qrwoq!(view(work_arr, 1:n, 1:(n+m)))

    yidx, xidx = 1:m, (m+1):(n+m)
    copy!(rsqrt(S), UpperTriangular(view(work_arr, yidx, yidx)))
    copy!(rsqrt(Σ), UpperTriangular(view(work_arr, xidx, xidx)))

    Kadj = view(work_arr, yidx, xidx)
    copy!(K, adjoint(Kadj))
    return S, K, Σ
end

function _schur_reduce!(
    Σ::Cholesky,
    K::AbstractVector,
    Π::Cholesky,
    C::Adjoint{<:Number,<:AbstractVector},
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    m, n = size(C)
    work_arr = view(work_arr, 1:(n+m), 1:(n+m))

    mul!(view(work_arr, 1:n, 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, 1:n, (m+1):(n+m)) .= rsqrt(Π)
    view(work_arr, (n+1):(n+m), 1:(n+m)) .= zero(eltype(work_arr))
    positive_qrwoq!(view(work_arr, 1:n, 1:(n+m)))

    yidx, xidx = 1, (m+1):(n+m) # yidx = 1:m = 1:1 but set to 1 so relevant SubArrays become vectors

    S = abs2(work_arr[1, 1])
    copy!(rsqrt(Σ), UpperTriangular(view(work_arr, xidx, xidx)))

    Kadj = view(work_arr, yidx, xidx)
    Kadj = conj!(Kadj) # view gives Kadj as a vector so only conjugation necessary
    copy!(K, Kadj)

    return S, K, Σ
end

# to be deleted (breaks htransform_and_likelihood ?)
function _schur_reduce(Π::Cholesky, C::AbstractMatrix, R)
    m, n = size(C)
    work_arr = similar(C, n + m, n + m)

    S = psdsimilar(Π, m)
    K = similar(adjoint(C))
    Σ = psdsimilar(Π, n)
    return _schur_reduce!(S, K, Σ, Π, C, R, work_arr)
end

# to be deleted (breaks htransform_and_likelihood ?)
function _schur_reduce(Π::Cholesky, C::Adjoint{<:Number,<:AbstractVector}, R::Number)
    m, n = size(C)
    work_arr = similar(C, n + m, n + m)

    K = similar(adjoint(C))
    Σ = psdsimilar(Π, n)
    return _schur_reduce!(K, Σ, Π, C, R, work_arr)
end

# to be deleted (breaks htransform_and_likelihood ?)
function _schur_reduce(
    Π::Cholesky,
    C::Adjoint{<:Number,<:AbstractVector},
    R::UniformScaling,
)
    return _schur_reduce(Π, C, R.λ)
end

function schur_reduce(
    Π::Cholesky,
    C::AbstractMatrix,
    R,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    m, n = size(C)
    S = psdsimilar(Π, m)
    K = similar(adjoint(C))
    Σ = psdsimilar(Π, n)
    return schur_reduce!(S, K, Σ, Π, C, R, work_arr)
end

function schur_reduce(
    Π::Cholesky,
    C::Adjoint{<:Number,<:AbstractVector},
    R::Number,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    m, n = size(C)
    K = similar(adjoint(C))
    Σ = psdsimilar(Π, n)
    return schur_reduce!(K, Σ, Π, C, R, work_arr)
end

function schur_reduce(
    Π::Cholesky,
    C::Adjoint{<:Number,<:AbstractVector},
    R::UniformScaling,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    return schur_reduce(Π, C, R.λ, work_arr)
end

function schur_reduce!(
    S::Cholesky,
    K::AbstractMatrix,
    Σ::Cholesky,
    Π::Cholesky,
    C::AbstractMatrix,
    R,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    S, K, Σ = _schur_reduce!(S, K, Σ, Π, C, R, work_arr)
    K = rdiv!(K, lsqrt(S))
    return S, K, Σ
end

function schur_reduce!(
    K::AbstractVector,
    Σ::Cholesky,
    Π::Cholesky,
    C::Adjoint{<:Number,<:AbstractVector},
    R::Number,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    S, K, Σ = _schur_reduce!(K, Σ, Π, C, R, work_arr)
    K = rdiv!(K, lsqrt(S))
    return S, K, Σ
end

function schur_reduce!(
    K::AbstractVector,
    Σ::Cholesky,
    Π::Cholesky,
    C::Adjoint{<:Number,<:AbstractVector},
    R::UniformScaling,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    return schur_reduce!(K, Σ, Π, C, R.λ, work_arr)
end

function _schur_reduce!(
    S::Cholesky,
    K::AbstractMatrix,
    Σ::Cholesky,
    Π::Cholesky,
    C::AbstractMatrix,
    R,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    m, n = size(C)
    work_arr = view(work_arr, 1:(n+m), 1:(n+m))
    copyto!(view(work_arr, 1:m, 1:m), rsqrt(R))
    view(work_arr, 1:m, (m+1):(n+m)) .= zero(eltype(work_arr))
    mul!(view(work_arr, (m+1):(n+m), 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, (m+1):(n+m), (m+1):(n+m)) .= rsqrt(Π)
    positive_qrwoq!(work_arr)

    yidx, xidx = 1:m, (m+1):(n+m)
    copy!(rsqrt(S), UpperTriangular(work_arr[yidx, yidx]))
    copy!(rsqrt(Σ), UpperTriangular(work_arr[xidx, xidx]))

    Kadj = @inbounds view(work_arr, yidx, xidx)
    copy!(K, adjoint(Kadj))

    return S, K, Σ
end

function _schur_reduce!(
    K::AbstractVector,
    Σ::Cholesky,
    Π::Cholesky,
    C::Adjoint{<:Number,<:AbstractVector},
    R::Number,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    m, n = size(C) # m = 1
    work_arr = view(work_arr, 1:(m+n), 1:(m+n))

    view(work_arr, 1:m, 1:m) .= rsqrt(R)
    view(work_arr, 1:m, (m+1):(n+m)) .= zero(eltype(work_arr))
    mul!(view(work_arr, (m+1):(n+m), 1:m), rsqrt(Π), adjoint(C))
    view(work_arr, (m+1):(n+m), (m+1):(n+m)) .= rsqrt(Π)
    positive_qrwoq!(work_arr)

    yidx, xidx = 1, (m+1):(n+m) # yidx = 1:m = 1:1 but set to 1 so relevant SubArrays become vectors

    S = abs2(work_arr[1, 1])
    copy!(rsqrt(Σ), UpperTriangular(work_arr[xidx, xidx]))

    Kadj = view(work_arr, yidx, xidx)
    Kadj = conj!(Kadj) # view gives Kadj as a vector so only conjugation necessary
    copy!(K, Kadj)

    return S, K, Σ
end

function _schur_reduce!(
    K::AbstractVector,
    Σ::Cholesky,
    Π::Cholesky,
    C::Adjoint{<:Number,<:AbstractVector},
    R::UniformScaling,
    work_arr::AbstractMatrix = similar(C, sum(size(C)), sum(size(C))),
)
    return _schur_reduce!(K, Σ, Π, C, R.λ, work_arr)
end

#=
function schur_reduce(Π::Cholesky, C::AbstractMatrix, R)
    S, K, Σ = _schur_reduce(Π, C, R)
    K = rdiv!(K, lsqrt(S))
    return S, K, Σ
end
=#
