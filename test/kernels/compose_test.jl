function compose_test(T, n, affine_types, cov_type)


    dirac_slopes, dirac_intercepts, dirac_amaps =
        collect(zip(map(x -> _make_affinemap(T, n, n, x), affine_types)...))
    dirac_kernels = collect(map(x -> DiracKernel(x), dirac_amaps))

    normal_amaps, cov_mats, cov_params, normal_kernels =
        collect(zip(map(x -> _make_normalkernel(T, n, n, x, cov_type), affine_types)...))

    x = randn(T, n)

    for i in 1:length(dirac_kernels), j in 1:length(dirac_kernels)
        K2 = dirac_kernels[j]
        F2 = dirac_amaps[j]
        K1 = dirac_kernels[i]
        F1 = dirac_amaps[i]

        @testset "compose | $(typeof(K2)) | $(typeof(K1))" begin
            @test mean(compose(K2, K1)) == compose(F2, F1)
            @test mean(compose(K1, K2)) == compose(F1, F2)
        end
    end

    for i in 1:length(normal_kernels), j in 1:length(dirac_kernels)
        K2 = dirac_kernels[j]
        F2 = dirac_amaps[j]
        K1 = normal_kernels[i]
        F1 = normal_amaps[i]
        cov_mat1 = cov_mats[i]
        @testset "compose | $(typeof(K2)) | $(typeof(K1))" begin
            @test mean(compose(K2, K1)) == compose(F2, F1)
            @test cov(condition(compose(K2, K1), x)) ≈
                  slope(mean(K2)) * cov_mat1 * slope(mean(K2))'
        end
    end

    for i in 1:length(dirac_kernels), j in 1:length(normal_kernels)
        K1 = dirac_kernels[i]
        F1 = dirac_amaps[i]
        K2 = normal_kernels[j]
        F2 = normal_amaps[j]
        cov_mat2 = cov_mats[j]
        @testset "compose | $(typeof(K2)) | $(typeof(K1))" begin
            @test mean(compose(K2, K1)) == compose(F2, F1)
            @test cov(condition(compose(K2, K1), x)) ≈ cov_mat2
        end
    end

    for i in 1:length(normal_kernels), j in 1:length(normal_kernels)
        K1 = normal_kernels[i]
        F1 = normal_amaps[i]
        cov_mat1 = cov_mats[i]
        K2 = normal_kernels[j]
        F2 = normal_amaps[j]
        cov_mat2 = cov_mats[j]
        @testset "compose | $(typeof(K2)) | $(typeof(K1))" begin
            @test mean(compose(K2, K1)) == compose(F2, F1)
            @test cov(condition(compose(K2, K1), x)) ≈
                  slope(mean(K2)) * cov_mat1 * slope(mean(K2))' + cov_mat2
        end
    end
end
