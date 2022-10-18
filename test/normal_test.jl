function normal_test(T, n, cov_types)
    ncovps = length(cov_types)

    means, cov_matrices, cov_parameters, normals = _make_normals(T, n, cov_types)

    x = randn(T, n)

    eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    for i in 1:ncovps
        @testset "Normal | Unary | $(T) | $(cov_types[i])" begin
            N = normals[i]
            μ = means[i]
            covmat = cov_matrices[i]
            covpar = cov_parameters[i]

            @test eltype(N) == T
            @test convert(typeof(N), N) == N
            for U in eltypes
                eltype(AbstractNormal{U}(N)) == U
                convert(AbstractNormal{U}, N) == AbstractNormal{U}(N)
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

    means2, cov_matrices2, cov_parameters2, normals2 = _make_normals(T, n, cov_types)

    for i in 1:ncovps, j in i:ncovps
        @testset "Normal | Binary | $(T) | $(cov_types[i]) / $(cov_types[j])" begin
            N1 = normals[i]
            μ1 = means[i]
            covmat1 = cov_matrices[i]
            covpar1 = cov_parameters[i]

            N2 = normals2[j]
            μ2 = means2[j]
            covmat2 = cov_matrices2[j]
            covpar2 = cov_parameters2[j]

            @test kldivergence(N1, N2) ≈ _kld(T, μ1, covmat1, μ2, covmat2)
            @test kldivergence(N2, N1) ≈ _kld(T, μ2, covmat2, μ1, covmat1)
            @test eltype(kldivergence(N1, N2)) <: Real
            @test eltype(kldivergence(N2, N1)) <: Real
        end
    end
end
