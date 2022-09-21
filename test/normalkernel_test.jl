function normalkernel_test(T, n)

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
        @test eltype(K1) == T
        @test typeof(K1) <: AffineNormalKernel

        @test mean(K1)(x) ≈ Φ1 * x
        @test cov(K1) == Q1

        @test condition(K1, x) == Normal(Φ1 * x, Q1)

        @test slope(mean(compose(K2, K1))) == slope(mean(K2 * K1)) ≈ slope(mean(K3))
        @test cov(compose(K2, K1)) == cov(K2 * K1) ≈ cov(K3)
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
        @test cov(KC1) ≈ cov(K_gt1)
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
        @test cov(KC2) ≈ cov(K_gt2)
        @test slope(mean(KC2)) ≈ slope(mean(K_gt2))
        @test intercept(mean(KC2)) ≈ intercept(mean(K_gt2))
    end

    λ2 = 3.0
    λ3 = 1.0 / 2.0

    IK1 = NormalKernel(Φ1, λ2 * I)
    IK2 = NormalKernel(Φ2, λ3 * I)

    K4 = NormalKernel(Φ2 * Φ1, Φ2 * λ2 * Φ2' + λ3 * I)

    @testset "AffineIsoNormalKernel | $(T) " begin
        @test mean(IK1)(x) == Φ1 * x
        @test cov(IK1) == λ2 * I

        @test condition(IK1, x) == Normal(Φ1 * x, λ2 * I)

        @test slope(mean(compose(IK2, IK1))) == slope(mean(IK2 * IK1)) ≈ slope(mean(K4))
        @test cov(compose(IK2, IK1)) == cov(IK2 * IK1) ≈ cov(K4)
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
        @test cov(KC3) ≈ cov(K_gt3)
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
        @test cov(KC4) ≈ cov(K_gt4)
        @test slope(mean(KC4)) ≈ slope(mean(K_gt4))
        @test intercept(mean(KC4)) ≈ intercept(mean(K_gt4))
    end
end

function _make_normal(T, n)
    RV = randn(T, n, n)
    Σ = Hermitian(RV' * RV)
    μ = randn(T, n)

    return μ, Σ, Normal(μ, Σ)
end

function _schur(Σ, μ, C, R)
    pred = C * μ
    # dimx = length(μ)
    # dimy = length(pred)

    S = Hermitian(C * Σ * C' + R)
    G = Σ * C' / S
    Π = Hermitian(Σ - G * S * G')

    return pred, S, G, Π
end
