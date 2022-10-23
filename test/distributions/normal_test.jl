function normal_test(T, n, cov_types)
    ncovps = length(cov_types)

    means, ncov_mats, ncov_params, normals =
        collect(zip(map(x -> _make_normal(T, n, x), cov_types)...))

    x = randn(T, n)

    eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    @testset "Normal | AbstractMatrix constructor" begin
        @test_throws DomainError Normal(ones(2), tril(ones(2, 2)))
        @test_throws DomainError Normal(ones(ComplexF64, 2), tril(ones(2, 2)))
    end

    @testset "Normal | Unary | $(T)" begin
        @test IsoNormal(x, one(real(T))) == Normal(x, one(T) * I)

        for i in 1:ncovps
            N = normals[i]
            μ = means[i]
            covmat = ncov_mats[i]
            covpar = ncov_params[i]

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
            covpar1 = ncov_params[i]

            N2 = normals2[j]
            μ2 = means2[j]
            covmat2 = ncov_mats2[j]
            covpar2 = ncov_params2[j]

            @test kldivergence(N1, N2) ≈ _kld(T, μ1, covmat1, μ2, covmat2)
            @test kldivergence(N2, N1) ≈ _kld(T, μ2, covmat2, μ1, covmat1)
            @test eltype(kldivergence(N1, N2)) <: Real
            @test eltype(kldivergence(N2, N1)) <: Real
        end
    end
end
