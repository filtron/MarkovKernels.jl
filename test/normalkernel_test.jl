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

    pred = Φ1 * μ
    S = Hermitian(Φ1 * Σ * Φ1' + Q1)
    G = Σ * Φ1' / S
    N_gt = Normal(pred, S)
    Π = Hermitian(Σ - G * S * G')

    corrector = AffineMap(G, μ, pred)
    K_gt = NormalKernel(corrector, Π)

    Nc, Kc = invert(N1, K1)

    @testset "NormalKernel | $(T)" begin
        @test eltype(K1) == T

        @test mean(K1)(x) ≈ Φ1 * x
        @test cov(K1) == Q1

        @test condition(K1, x) == N12

        @test slope(mean(compose(K2, K1))) ≈ slope(K3.μ)
        @test cov(compose(K2, K1)) ≈ cov(K3)
        @test slope(mean(K2 * K1)) == slope(mean(compose(K2, K1)))
        @test cov(K2 * K1) == cov(compose(K2, K1))

        @test mean(marginalise(N1, K1)) ≈ Φ1 * μ
        @test cov(marginalise(N1, K1)) ≈ Hermitian(Φ1 * Σ * Φ1' + Q1)

        @test mean(Nc) ≈ mean(N_gt)
        @test cov(Nc) ≈ cov(N_gt)
        @test cov(Kc) ≈ cov(K_gt)
        @test slope(mean(Kc)) ≈ slope(mean(K_gt))
        @test intercept(mean(Kc)) ≈ intercept(mean(K_gt))
    end
end
