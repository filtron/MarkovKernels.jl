function likelihood_test(T, n, m, affine_types, cov_type)
    dirac_slopes, dirac_intercepts, dirac_amaps =
        collect(zip(map(x -> _make_affinemap(T, n, m, x), affine_types)...))
    dirac_kernels = collect(map(x -> DiracKernel(x), dirac_amaps))

    # normal_kernel_types = Iterators.product(affine_types, cov_types)
    # normal_amaps, kcov_mats, kcov_params, normal_kernels =
    #     collect(zip(map(x -> _make_normalkernel(T, n, m, x...), normal_kernel_types)...))

    # means, ncov_mats, ncov_params, normals =
    #     collect(zip(map(x -> _make_normal(T, m, x), cov_types)...))

    normal_amaps, kcov_mats, kcov_params, normal_kernels =
        collect(zip(map(x -> _make_normalkernel(T, n, m, x, cov_type), affine_types)...))

    μ, ncovm, ncov_params, N = _make_normal(T, m, cov_type)

    y = randn(T, n)
    x = randn(T, m)

    normal_loglikes =
        map(x -> LogLike(x...), zip(normal_kernels, fill(y, length(normal_kernels))))
    dirac_loglikes =
        map(x -> LogLike(x...), zip(dirac_kernels, fill(y, length(dirac_kernels))))

    @testset "LogLike | AffineNormal" begin
        for i in 1:length(normal_loglikes)
            L = normal_loglikes[i]
            K = normal_kernels[i]
            @test L == LogLike(K, y)
            @test measurement(L) == y
            @test measurement_model(L) == K
            @test L(x) ≈ logpdf(condition(K, x), y)
        end
    end

    @testset "LogLike | AffineDirac" begin
        for i in 1:length(dirac_loglikes)
            L = dirac_loglikes[i]
            K = dirac_kernels[i]
            @test L == LogLike(K, y)
            @test measurement(L) == y
            @test measurement_model(L) == K
        end
    end

    @testset "Loglike | AffineNormal | bayes_rule" begin
        for j in 1:length(normal_kernels)
            L = normal_loglikes[j]
            K = normal_kernels[j]

            M, KC = invert(N, K)

            NC, loglike = bayes_rule(N, L)
            @test mean(NC) ≈ mean(condition(KC, y))
            @test cov(NC) ≈ cov(condition(KC, y))
            @test loglike ≈ logpdf(M, y)
        end
    end

    @testset "Loglike | AffineDirac | bayes_rule" begin
        for j in 1:length(dirac_kernels)
            L = dirac_loglikes[j]
            K = dirac_kernels[j]

            M, KC = invert(N, K)

            NC, loglike = bayes_rule(N, L)
            @test mean(NC) ≈ mean(condition(KC, y))
            @test cov(NC) ≈ cov(condition(KC, y))
            @test loglike ≈ logpdf(M, y)
        end
    end
end
