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

            @test cov(condition(compose(DK, NK), x)) ≈ slope(mean(DK))*cov_mat*slope(mean(DK))'
            @test cov(condition(compose(NK, DK), x)) ≈ cov_mat
        end

    end




    RV = randn(T, m, m)
    Σ = Hermitian(RV' * RV)
    μ = randn(T, m)
    N1 = Normal(μ, Σ)

    Φ1 = randn(T, n, m)
    K1 = DiracKernel(Φ1)


    Φ2 = randn(T, m, n)
    K2 = DiracKernel(Φ2)

    K3 = DiracKernel(Φ2 * Φ1)

    x = randn(T, m)

    D12 = Dirac(Φ1 * x)

    pred = Φ1 * μ
    S = Hermitian(Φ1 * Σ * Φ1')
    G = Σ * Φ1' / S
    N_gt = Normal(pred, S)
    Π = Hermitian(Σ - G * S * G')

    corrector = AffineCorrector(G, μ, pred)
    K_gt = NormalKernel(corrector, Π)

    Nc, Kc = invert(N1, K1)

    @testset "DiracKernel | $(T)" begin
        @test eltype(K1) == T

        @test mean(K1)(x) ≈ Φ1 * x
        @test cov(K1)(x) == Diagonal(zeros(T, size(Φ1, 1)))

        @test condition(K1, x) == D12

        @test slope(mean(compose(K2, K1))) ≈ slope(K3.μ)
        @test cov(compose(K2, K1))(x) ≈ cov(K3)(x)

        @test mean(marginalise(N1, K1)) ≈ Φ1 * μ
        @test cov(marginalise(N1, K1)) ≈ Hermitian(Φ1 * Σ * Φ1')

        @test mean(Nc) ≈ mean(N_gt)
        @test cov(Nc) ≈ cov(N_gt)
        @test cov(Kc)(x) ≈ cov(K_gt)(x)
        @test slope(mean(Kc)) ≈ slope(mean(K_gt))
        @test intercept(mean(Kc)) ≈ intercept(mean(K_gt))
    end
end
