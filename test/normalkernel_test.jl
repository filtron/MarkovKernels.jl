function normalkernel_test(T, n)

    # define a normal distribution

    RV = randn(T, n, n)
    Σ = Hermitian(RV' * RV)
    μ = randn(T, n)
    N1 = Normal(μ, Σ)

    Φ1 = randn(T, n, n)
    RQ1 = randn(T, n, n)
    Q1 = Hermitian(RQ1' * RQ1)
    K1 = NormalKernel(Φ1, Q1)

    Φ2 = randn(T, n, n)
    RQ2 = randn(T, n, n)
    Q2 = Hermitian(RQ2' * RQ2)
    K2 = NormalKernel(Φ2, Q2)

    K3 = NormalKernel(Φ2 * Φ1, Hermitian(Φ2 * Q1 * Φ2' + Q2))

    x = randn(T, n)

    N12 = Normal(Φ1 * x, Q1)

    pred, S, G, Π = _schur(Σ, μ, Φ1, Q1)
    N_gt1 = Normal(pred, S)
    corrector = AffineMap(G, μ, pred)
    K_gt1 = NormalKernel(corrector, Π)

    NC1, KC1 = invert(N1, K1)

    @testset "AffineNormalKernel | $(T)" begin
        @test eltype(K1) == T
        @test typeof(K1) <: AffineNormalKernel

        @test mean(K1)(x) ≈ Φ1 * x
        @test cov(K1) == Q1

        @test condition(K1, x) == N12

        @test slope(mean(compose(K2, K1))) ≈ slope(K3.μ)
        @test cov(compose(K2, K1)) ≈ cov(K3)
        @test slope(mean(K2 * K1)) == slope(mean(compose(K2, K1)))
        @test cov(K2 * K1) == cov(compose(K2, K1))

        @test mean(marginalise(N1, K1)) ≈ Φ1 * μ
        @test cov(marginalise(N1, K1)) ≈ Hermitian(Φ1 * Σ * Φ1' + Q1)

        @test mean(NC1) ≈ mean(N_gt1)
        @test cov(NC1) ≈ cov(N_gt1)
        @test cov(KC1) ≈ cov(K_gt1)
        @test slope(mean(KC1)) ≈ slope(mean(K_gt1))
        @test intercept(mean(KC1)) ≈ intercept(mean(K_gt1))
    end

    λ1 = 2.0
    IN1 = IsoNormal(μ,λ1)

    pred, S, G, Π = _schur(λ1*I, μ, Φ1, Q1)
    N_gt2 = Normal(pred, S)
    corrector = AffineMap(G, μ, pred)
    K_gt2 = NormalKernel(corrector, Π)

    NC2, KC2 = invert(IN1, K1)

    @testset "AffineNormalKernel / IsoNormal | $(T) " begin

        @test mean(marginalise(IN1, K1)) ≈ Φ1 * μ
        @test cov(marginalise(IN1, K1)) ≈ Hermitian(Φ1 * λ1 * Φ1' + Q1)

        @test mean(NC2) ≈ mean(N_gt2)
        @test cov(NC2) ≈ cov(N_gt2)
        @test cov(KC2) ≈ cov(K_gt2)
        @test slope(mean(KC2)) ≈ slope(mean(K_gt2))
        @test intercept(mean(KC2)) ≈ intercept(mean(K_gt2))
    end

end

function _schur(Σ, μ, C,R)

    pred = C * μ
  #  dimx = length(μ)
   # dimy = length(pred)

    S = Hermitian(C * Σ * C' + R)
    G = Σ * C' / S
    Π = Hermitian(Σ - G * S * G')

    return pred, S, G, Π
end