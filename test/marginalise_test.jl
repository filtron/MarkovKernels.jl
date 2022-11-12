function marginalise_test(T, n, m, affine_types, cov_type)
    dirac_slopes, dirac_intercepts, dirac_amaps =
        collect(zip(map(x -> _make_affinemap(T, n, m, x), affine_types)...))
    dirac_kernels = collect(map(x -> DiracKernel(x), dirac_amaps))

    normal_amaps, kcov_mats, kcov_params, normal_kernels =
        collect(zip(map(x -> _make_normalkernel(T, n, m, x, cov_type), affine_types)...))

    dirac = Dirac(randn(T, m))

    μ, ncovm, ncov_params, D = _make_normal(T, m, cov_type)

    for j in 1:length(normal_kernels)
        K = normal_kernels[j]
        kcovm = kcov_mats[j]
        F = normal_amaps[j]
        @testset "marginalise | $(typeof(D)) | $(typeof(K))" begin
            @test mean(marginalise(D, K)) ≈ F(μ)
            @test cov(marginalise(D, K)) ≈ slope(F) * ncovm * slope(F)' + kcovm
        end
    end

    for j in 1:length(dirac_kernels)
        K = dirac_kernels[j]
        F = dirac_amaps[j]
        @testset "marginalise | $(typeof(D)) | $(typeof(K))" begin
            @test mean(marginalise(D, K)) ≈ F(μ)
            @test cov(marginalise(D, K)) ≈ slope(F) * ncovm * slope(F)'
        end
    end

    for j in 1:length(normal_kernels)
        D = dirac
        μ = mean(D)

        K = normal_kernels[j]
        kcovm = kcov_mats[j]
        F = normal_amaps[j]
        @testset "marginalise | $(typeof(D)) | $(typeof(K))" begin
            @test mean(marginalise(D, K)) ≈ F(μ)
            @test cov(marginalise(D, K)) ≈ kcovm
        end
    end

    for j in 1:length(dirac_kernels)
        D = dirac
        μ = mean(D)

        K = dirac_kernels[j]
        F = dirac_amaps[j]
        @testset "marginalise | $(typeof(D)) | $(typeof(K))" begin
            @test mean(marginalise(D, K)) ≈ F(μ)
        end
    end
end
