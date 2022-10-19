function dirackernel_test(T, n, affine_types, cov_types)
    for at in affine_types
        slope, intercept, F = _make_affinemap(T, n, n, at)
        K = DiracKernel(F)
        x = randn(T, n)

        eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

        @testset "AffineDiracKernel | Unary | $(T) | $(at)" begin
            @test eltype(K) == T
            @test typeof(K) <: AffineDiracKernel
            @test convert(typeof(K), K) == K
            for U in eltypes
                eltype(AbstractDiracKernel{U}(K)) == U
                convert(AbstractDiracKernel{U}, K) == AbstractDiracKernel{U}(K)
            end
            @test mean(K)(x) == F(x)
            @test cov(K)(x) == Diagonal(zeros(T, n))
            @test condition(K, x) == Dirac(F(x))
        end
    end

    for dk_at in affine_types, n_ct in cov_types
        _slope, _intercept, DF = _make_affinemap(T, n - 1, n, dk_at)
        DK = DiracKernel(DF)
        m, ncov_mat, ncov_param, N = _make_normal(T, n, n_ct)

        NC, KC = invert(N, DK)
        x = randn(T, n - 1)

        pred, S, G, Π = _schur(ncov_mat, m, slope(DF)) # _schur should not return pred
        pred = DF(m)
        Ngt = Normal(pred, S)
        Kgt = NormalKernel(G, m, pred, Π)

        @testset "AffineDiracKernel {$(T),$(dk_at)} | Normal {$(T),$(n_ct)}" begin
            @test mean(marginalise(N, DK)) ≈ DF(m)
            @test cov(marginalise(N, DK)) ≈ slope(DF) * ncov_mat * slope(DF)'

            @test mean(NC) ≈ mean(Ngt)
            @test cov(NC) ≈ cov(Ngt)

            @test slope(mean(KC)) ≈ slope(mean(Kgt))
            @test intercept(mean(KC)) ≈ intercept(mean(Kgt))
            @test cov(condition(KC, x)) ≈ cov(condition(Kgt, x))
            @test mean(condition(KC, x)) ≈ mean(condition(Kgt, x))
        end
    end

    normal_kernel_type_parameters = Iterators.product(affine_types, cov_types)

    for dk_at in affine_types, nkt in normal_kernel_type_parameters
        nk_at, nk_ct = nkt

        _slope, _intercept, DF = _make_affinemap(T, n, n, dk_at)
        DK = DiracKernel(DF)
        NF, cov_mat, cov_param, NK = _make_normalkernel(T, n, n, nk_at, nk_ct)

        x = randn(T, n)

        @testset "AffineDiracKernel {$(T),$(dk_at)} | AffineNormaKernel {$(T),$(nk_at),$(nk_ct)}" begin
            @test mean(compose(DK, NK)) == compose(DF, NF)
            @test mean(compose(NK, DK)) == compose(NF, DF)

            @test cov(condition(compose(DK, NK), x)) ≈
                  slope(mean(DK)) * cov_mat * slope(mean(DK))'
            @test cov(condition(compose(NK, DK), x)) ≈ cov_mat
        end
    end
end
