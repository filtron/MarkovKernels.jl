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
    elseif s === :UniformScaling
        λ = abs2.(randn(T))
        covariance_parameter = UniformScaling(λ)
        covariance_matrix = covariance_parameter(n)
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

function _make_normals(T, n, cov_types)
    means = []
    cov_matrices = []
    cov_parameters = []
    normals = []

    for i in 1:length(cov_types)
        mean, cov_matrix, cov_param, normal = _make_normal(T, n, cov_types[i])
        push!(means, mean)
        push!(cov_matrices, cov_matrix)
        push!(cov_parameters, cov_param)
        push!(normals, normal)
    end

    return means, cov_matrices, cov_parameters, normals
end

function _make_normalkernel(T, n, m, atype::Symbol, ctype::Symbol)
    cov_param, cov_mat = _make_cov(T, n, ctype)
    slope, intercept, M = _make_affinemap(T, n, m, atype)
    K = NormalKernel(M, cov_param)
    return M, cov_mat, cov_param, K
end

_symmetrise(T, Σ, n) = Σ
_symmetrise(T, Σ::Diagonal, n) = Σ
_symmetrise(T, Σ::UniformScaling, n) = Σ(n)
_symmetrise(T, Σ::AbstractMatrix) = T <: Real ? Symmetric(Σ) : Hermitian(Σ)

function _logpdf(T, μ1, Σ1, x1)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1, n)
    if T <: Real
        logpdf = -0.5 * logdet(2 * π * Σ1) - 0.5 * dot(x1 - μ1, inv(Σ1), x1 - μ1)
    elseif T <: Complex
        logpdf = -n * log(π) - logdet(Σ1) - dot(x1 - μ1, inv(Σ1), x1 - μ1)
    end

    return logpdf
end

function _entropy(T, μ1, Σ1)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1, n)
    if T <: Real
        entropy = 1.0 / 2.0 * logdet(2 * π * exp(1) * Σ1)
    elseif T <: Complex
        entropy = n * log(π) + logdet(Σ1) + n
    end
end

function _kld(T, μ1, Σ1, μ2, Σ2)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1, n)
    Σ2 = _symmetrise(T, Σ2, n)

    if T <: Real
        kld =
            1 / 2 *
            (tr(Σ2 \ Σ1) - n + dot(μ2 - μ1, inv(Σ2), μ2 - μ1) + logdet(Σ2) - logdet(Σ1))
    elseif T <: Complex
        kld =
            real(tr(Σ2 \ Σ1)) - n + real(dot(μ2 - μ1, inv(Σ2), μ2 - μ1)) + logdet(Σ2) -
            logdet(Σ1)
    end

    return kld
end

# this should call symmetrise
function _schur(Σ, μ, C, R)
    pred = C * μ

    S = Hermitian(C * Σ * C' + R)
    G = Σ * C' / S
    Π = Hermitian(Σ - G * S * G')

    return pred, S, G, Π
end
