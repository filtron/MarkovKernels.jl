function invert_test(T, n, m, affine_types, cov_types)
    dirac_slopes, dirac_intercepts, dirac_amaps =
        collect(zip(map(x -> _make_affinemap(T, n, m, x), affine_types)...))
    dirac_kernels = collect(map(x -> DiracKernel(x), dirac_amaps))

    normal_kernel_types = Iterators.product(affine_types, cov_types)
    normal_amaps, kcov_mats, kcov_params, normal_kernels =
        collect(zip(map(x -> _make_normalkernel(T, n, m, x...), normal_kernel_types)...))

    means, ncov_mats, ncov_params, normals =
        collect(zip(map(x -> _make_normal(T, m, x), cov_types)...))

    for i in 1:length(normals), j in 1:length(normal_kernels)
        D = normals[i]
        μ = means[i]
        ncovm = ncov_mats[i]

        K = normal_kernels[j]
        kcovm = kcov_mats[j]
        F = normal_amaps[j]

        NC, KC = invert(D, K)
        y = randn(T, n)

        S, G, Π = _schur(ncovm, slope(F), kcovm) # _schur should not return pred
        pred = F(μ)
        Ngt = Normal(pred, S)
        Kgt = NormalKernel(G, μ, pred, Π)

        @testset "invert | $(typeof(D)) | $(typeof(K))" begin
            @test mean(NC) ≈ mean(Ngt)
            @test cov(NC) ≈ cov(Ngt)

            @test slope(mean(KC)) ≈ slope(mean(Kgt))
            @test intercept(mean(KC)) ≈ intercept(mean(Kgt))
            @test cov(condition(KC, y)) ≈ cov(condition(Kgt, y))
            @test mean(condition(KC, y)) ≈ mean(condition(Kgt, y))
        end
    end

    for i in 1:length(normals), j in 1:length(dirac_kernels)
        D = normals[i]
        μ = means[i]
        ncovm = ncov_mats[i]

        K = dirac_kernels[j]
        F = dirac_amaps[j]

        NC, KC = invert(D, K)
        y = randn(T, n)

        S, G, Π = _schur(ncovm, slope(F)) # _schur should not return pred
        pred = F(μ)
        Ngt = Normal(pred, S)
        Kgt = NormalKernel(G, μ, pred, Π)

        @testset "invert | $(typeof(D)) | $(typeof(K))" begin
            @test mean(NC) ≈ mean(Ngt)
            @test cov(NC) ≈ cov(Ngt)

            @test slope(mean(KC)) ≈ slope(mean(Kgt))
            @test intercept(mean(KC)) ≈ intercept(mean(Kgt))
            @test cov(condition(KC, y)) ≈ cov(condition(Kgt, y))
            @test mean(condition(KC, y)) ≈ mean(condition(Kgt, y))
        end
    end
end
