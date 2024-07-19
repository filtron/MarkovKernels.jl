function normal_test(T, n, cov_types, matrix_types)
    @testset "Normal | AbstractMatrix constructor" begin
        @test_throws DomainError Normal(ones(2), tril(ones(2, 2)))
        @test_throws DomainError Normal(ones(ComplexF64, 2), tril(ones(2, 2)))
        @test_throws DomainError Normal(ones(ComplexF64, 2), Symmetric(diagm(ones(2))))
    end

    eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    xp = randn(T, n)
    μ1p = randn(T, n)
    V1p = tril(ones(T, n, n))
    V1p = V1p * V1p'

    for cov_t in cov_types, matrix_t in matrix_types
        x = _make_vector(xp, matrix_t)
        μ = _make_vector(μ1p, matrix_t)
        Σ = _make_covp(_make_matrix(V1p, matrix_t), cov_t)

        N = Normal(μ, Σ)

        @testset "Normal | Unary | $(T) | " begin
            @test_nowarn repr(N)
            @test eltype(N) == T

            @test !(copy(N) === N)
            @test copy(N) == N
            @test typeof(copy(N)) === typeof(N)
            @test typeof(similar(N)) === typeof(N)
            _N = similar(N)
            @test (copy!(_N, N); _N) == N

            @test !(recursivecopy(N) === N)
            @test recursivecopy(N) == N
            @test typeof(recursivecopy(N)) === typeof(N)
            _N = similar(N)
            @test (recursivecopy!(_N, N); _N) == N

            for U in eltypes
                @test AbstractDistribution{U}(N) == AbstractNormal{U}(N) == Normal{U}(N)
                @test eltype(AbstractNormal{U}(N)) == U
            end
            @test N == N
            @test mean(N) == μ
            @test cov(N) ≈ V1p
            @test covp(N) == Σ
            @test var(N) ≈ real.(diag(V1p))
            @test std(N) ≈ sqrt.(real.(diag(V1p)))

            @test residual(N, x) ≈ cholesky(V1p).L \ (x - μ)
            @test _ofsametype(x, residual(N, x))
            @test logpdf(N, x) ≈ _logpdf(T, μ, V1p, x)
            @test entropy(N) ≈ _entropy(T, μ, V1p)

            @test eltype(var(N)) <: Real
            @test eltype(std(N)) <: Real
            @test eltype(logpdf(N, x)) <: Real
            @test eltype(entropy(N)) <: Real

            @test length(rand(N)) == dim(N)
            @test eltype(rand(N)) == T
            @test typeof(rand(N)) == typeof_sample(N)
            @test eltype(rand(N)) == eltype_sample(N)
        end
    end

    μ2p = randn(T, n)
    V2p = triu(ones(T, n, n))
    V2p = V2p * V2p'

    for cov_t in cov_types, matrix_t in matrix_types
        x = _make_vector(xp, matrix_t)
        μ1 = _make_vector(μ1p, matrix_t)
        Σ1 = _make_covp(_make_matrix(V1p, matrix_t), cov_t)
        N1 = Normal(μ1, Σ1)
        μ2 = _make_vector(μ2p, matrix_t)
        Σ2 = _make_covp(_make_matrix(V2p, matrix_t), cov_t)
        N2 = Normal(μ2, Σ2)

        @testset "Normal | Binary | {$(T),$(cov_t),$(matrix_t)}" begin
            @test kldivergence(N1, N2) ≈ _kld(T, μ1p, V1p, μ2p, V2p)
            @test kldivergence(N2, N1) ≈ _kld(T, μ2p, V2p, μ1p, V1p)
            @test eltype(kldivergence(N1, N2)) <: Real
            @test eltype(kldivergence(N2, N1)) <: Real
        end
    end
end

function _logpdf(T, μ1, Σ1, x1)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1)
    if T <: Real
        logpdf = -T(0.5) * logdet(T(2π) * Σ1) - T(0.5) * dot(x1 - μ1, inv(Σ1), x1 - μ1)
    elseif T <: Complex
        logpdf = -real(T)(n) * log(real(T)(π)) - logdet(Σ1) - dot(x1 - μ1, inv(Σ1), x1 - μ1)
    end

    return logpdf
end

function _entropy(T, μ1, Σ1)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1)
    if T <: Real
        entropy = T(0.5) * logdet(T(2π) * exp(T(1)) * Σ1)
    elseif T <: Complex
        entropy = real(T)(n) * log(real(T)(π)) + logdet(Σ1) + real(T)(n)
    end
end

function _kld(T, μ1, Σ1, μ2, Σ2)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1)
    Σ2 = _symmetrise(T, Σ2)

    if T <: Real
        kld =
            T(0.5) *
            (tr(Σ2 \ Σ1) - T(n) + dot(μ2 - μ1, inv(Σ2), μ2 - μ1) + logdet(Σ2) - logdet(Σ1))
    elseif T <: Complex
        kld =
            real(tr(Σ2 \ Σ1)) - real(T)(n) +
            real(dot(μ2 - μ1, inv(Σ2), μ2 - μ1)) +
            logdet(Σ2) - logdet(Σ1)
    end

    return kld
end
