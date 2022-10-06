function normalkernel_test(T, n, affine_types, cov_types)
    numa = length(affine_types)
    numc = length(cov_types)

    x = randn(T, n)
    for i in 1:numa, j in 1:numc
        atype = affine_types[i]
        ctype = cov_types[j]

        M, cov_mat, cov_param, K = _make_normalkernel(T, n, n, atype, ctype)
        @testset "NormalKernel | Unary | $(T) | $(atype) | $(ctype)" begin
            @test eltype(K) == T
            @test typeof(K) <: AffineNormalKernel
            @test mean(K)(x) == M(x)
            @test cov(K)(x) == cov_param
            @test covp(K) == cov_param
            @test condition(K, x) == Normal(M(x), cov_param)
        end
    end

    # define a normal distribution
    Φ1 = randn(T, n, n)
    RQ1 = randn(T, n, n)
    Q1 = RQ1' * RQ1
    K1 = NormalKernel(Φ1, Q1)

    Φ2 = randn(T, n, n)
    RQ2 = randn(T, n, n)
    Q2 = RQ2' * RQ2
    K2 = NormalKernel(Φ2, Q2)

    K3 = NormalKernel(Φ2 * Φ1, Φ2 * Q1 * Φ2' + Q2)

    x = randn(T, n)

    @testset "AffineNormalKernel | $(T)" begin
        @test slope(mean(compose(K2, K1))) ≈ slope(mean(K3))
        @test cov(compose(K2, K1))(x) ≈ cov(K3)(x)
        @test covp(compose(K2, K1)) ≈ covp(K3)
    end

    μ, Σ, N1 = _make_normal(T, n)
    pred, S, G, Π = _schur(Σ, μ, Φ1, Q1)
    N_gt1 = Normal(pred, S)
    corrector = AffineMap(G, μ, pred)
    K_gt1 = NormalKernel(corrector, Π)

    NC1, KC1 = invert(N1, K1)

    @testset "AffineNormalKernel / Normal | $(T) " begin
        @test mean(marginalise(N1, K1)) ≈ Φ1 * μ
        @test cov(marginalise(N1, K1)) ≈ Φ1 * Σ * Φ1' + Q1

        @test mean(NC1) ≈ mean(N_gt1)
        @test cov(NC1) ≈ cov(N_gt1)
        @test cov(KC1)(x) ≈ cov(K_gt1)(x)

        @test slope(mean(KC1)) ≈ slope(mean(K_gt1))
        @test intercept(mean(KC1)) ≈ intercept(mean(K_gt1))
    end

    λ1 = 2.0
    IN1 = IsoNormal(μ, λ1)

    pred, S, G, Π = _schur(λ1 * I, μ, Φ1, Q1)
    N_gt2 = Normal(pred, S)
    corrector = AffineMap(G, μ, pred)
    K_gt2 = NormalKernel(corrector, Π)

    NC2, KC2 = invert(IN1, K1)

    @testset "AffineNormalKernel / IsoNormal | $(T) " begin
        @test mean(marginalise(IN1, K1)) ≈ Φ1 * μ
        @test cov(marginalise(IN1, K1)) ≈ Φ1 * λ1 * Φ1' + Q1

        @test mean(NC2) ≈ mean(N_gt2)
        @test cov(NC2) ≈ cov(N_gt2)
        @test cov(KC2)(x) ≈ cov(K_gt2)(x)
        @test slope(mean(KC2)) ≈ slope(mean(K_gt2))
        @test intercept(mean(KC2)) ≈ intercept(mean(K_gt2))
    end

    λ2 = 3.0
    λ3 = 1.0 / 2.0

    IK1 = NormalKernel(Φ1, λ2 * I)
    IK2 = NormalKernel(Φ2, λ3 * I)

    K4 = NormalKernel(Φ2 * Φ1, Φ2 * λ2 * Φ2' + λ3 * I)

    @testset "AffineIsoNormalKernel | $(T) " begin
        @test slope(mean(compose(IK2, IK1))) ≈ slope(mean(K4))
        @test cov(compose(IK2, IK1))(x) ≈ cov(K4)(x)
    end

    pred, S, G, Π = _schur(Σ, μ, Φ1, λ2 * I)
    N_gt3 = Normal(pred, S)
    corrector = AffineMap(G, μ, pred)
    K_gt3 = NormalKernel(corrector, Π)

    NC3, KC3 = invert(N1, IK1)

    @testset "AffineIsoNormalKernel / Normal | $(T) " begin
        @test mean(marginalise(N1, IK1)) ≈ Φ1 * μ
        @test cov(marginalise(N1, IK1)) ≈ Φ1 * Σ * Φ1' + λ2 * I

        @test mean(NC3) ≈ mean(N_gt3)
        @test cov(NC3) ≈ cov(N_gt3)
        @test cov(KC3)(x) ≈ cov(K_gt3)(x)
        @test slope(mean(KC3)) ≈ slope(mean(K_gt3))
        @test intercept(mean(KC3)) ≈ intercept(mean(K_gt3))
    end

    pred, S, G, Π = _schur(λ1 * I, μ, Φ1, λ2 * I)
    N_gt4 = Normal(pred, S)
    corrector = AffineMap(G, μ, pred)
    K_gt4 = NormalKernel(corrector, Π)

    NC4, KC4 = invert(IN1, IK1)

    @testset "AffineIsoNormalKernel / IsoNormal | $(T) " begin
        @test mean(marginalise(IN1, IK1)) ≈ Φ1 * μ
        @test cov(marginalise(IN1, IK1)) ≈ Φ1 * λ1 * Φ1' + λ2 * I

        @test mean(NC4) ≈ mean(N_gt4)
        @test cov(NC4) ≈ cov(N_gt4)
        @test cov(KC4)(x) ≈ cov(K_gt4)(x)
        @test slope(mean(KC4)) ≈ slope(mean(K_gt4))
        @test intercept(mean(KC4)) ≈ intercept(mean(K_gt4))
    end
end
