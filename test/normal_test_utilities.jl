# plain implementations of operations on Normal distributions / Normal kernels
function _make_cov(T, n, s::Symbol)
    if s === :Matrix
        R = randn(T, n, n)
        covariance_matrix = R' * R
        covariance_parameter =
            T <: Real ? Symmetric(covariance_matrix) : Hermitian(covariance_matrix)
        covariance_matrix = covariance_parameter
    elseif s === :Diagonal
        d = randn(T, n, n)
        covariance_matrix = Diagonal(abs2.(d))
        covariance_parameter = covariance_matrix
    elseif s === :Cholesky
        R = randn(T, n, n)
        covariance_matrix = R' * R
        covariance_parameter = cholesky(covariance_matrix)
    end

    return covariance_parameter, covariance_matrix
end

function _make_normal(T, n, s::Symbol)
    mean = randn(T, n)
    covariance_parameter, covariance_matrix = _make_cov(T, n, s)
    N = Normal(mean, covariance_parameter)

    return mean, covariance_matrix, covariance_parameter, N
end

function _make_normalkernel(T, n, m, atype::Symbol, ctype::Symbol)
    cov_param, cov_mat = _make_cov(T, n, ctype)
    slope, intercept, M = _make_affinemap(T, n, m, atype)
    K = NormalKernel(M, cov_param)
    return M, cov_mat, cov_param, K
end

_symmetrise(T, Σ) = Σ
_symmetrise(T, Σ::AbstractMatrix) = T <: Real ? Symmetric(Σ) : Hermitian(Σ)

function _schur(Σ, C)
    S = C * Σ * C'
    S = _symmetrise(eltype(S), S)
    G = Σ * C' / S
    Π = Σ - G * S * G'
    Π = _symmetrise(eltype(Π), Π)
    return S, G, Π
end

function _schur(Σ, C, R)
    S = C * Σ * C' + R
    S = _symmetrise(eltype(S), S)
    G = Σ * C' / S
    Π = Σ - G * S * G'
    Π = _symmetrise(eltype(Π), Π)
    return S, G, Π
end
