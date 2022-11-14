function normal_test(T, n, cov_types)
    ncovps = length(cov_types)
    means, ncov_mats, ncov_params, normals =
        collect(zip(map(x -> _make_normal(T, n, x), cov_types)...))

    x = randn(T, n)

    eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    @testset "Normal | AbstractMatrix constructor" begin
        @test_throws DomainError Normal(ones(2), tril(ones(2, 2)))
        @test_throws DomainError Normal(ones(ComplexF64, 2), tril(ones(2, 2)))
        @test_throws DomainError Normal(ones(ComplexF64, 2), Symmetric(diagm(ones(2))))
    end

    @testset "Normal | Unary | $(T)" begin
        for i in 1:ncovps
            N = normals[i]
            μ = means[i]
            covmat = ncov_mats[i]
            covpar = ncov_params[i]

            @test_nowarn repr(N)
            @test eltype(N) == T
            for U in eltypes
                @test AbstractDistribution{U}(N) == AbstractNormal{U}(N) == Normal{U}(N)
                @test eltype(AbstractNormal{U}(N)) == U
            end
            @test N == N
            @test mean(N) == μ
            @test cov(N) ≈ covmat
            @test covp(N) == covpar
            @test var(N) ≈ real.(diag(covmat))
            @test std(N) ≈ sqrt.(real.(diag(covmat)))

            @test residual(N, x) ≈ cholesky(covmat).L \ (x - μ)
            @test logpdf(N, x) ≈ _logpdf(T, μ, covmat, x)
            @test entropy(N) ≈ _entropy(T, μ, covmat)

            @test eltype(var(N)) <: Real
            @test eltype(std(N)) <: Real
            @test eltype(logpdf(N, x)) <: Real
            @test eltype(entropy(N)) <: Real

            @test length(rand(N)) == dim(N)
            @test eltype(rand(N)) == T
        end
    end

    means2, ncov_mats2, ncov_params2, normals2 =
        collect(zip(map(x -> _make_normal(T, n, x), cov_types)...))

    @testset "Normal | Binary | $(T)" begin
        for i in 1:ncovps, j in i:ncovps
            N1 = normals[i]
            μ1 = means[i]
            covmat1 = ncov_mats[i]

            N2 = normals2[j]
            μ2 = means2[j]
            covmat2 = ncov_mats2[j]

            @test kldivergence(N1, N2) ≈ _kld(T, μ1, covmat1, μ2, covmat2)
            @test kldivergence(N2, N1) ≈ _kld(T, μ2, covmat2, μ1, covmat1)
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
