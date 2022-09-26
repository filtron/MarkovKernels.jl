function dirackernel_test(T, n, m)

    # define a normal distribution

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

    corrector = AffineMap(G, μ, pred)
    K_gt = NormalKernel(corrector, Π)

    Nc, Kc = invert(N1, K1)

    @testset "DiracKernel | $(T)" begin
        @test eltype(K1) == T

        @test mean(K1)(x) ≈ Φ1 * x
        @test cov(K1) == zeros(T, size(Φ1, 1), size(Φ1, 1))

        @test condition(K1, x) == D12

        @test slope(mean(compose(K2, K1))) ≈ slope(K3.μ)
        @test cov(compose(K2, K1)) ≈ cov(K3)

        @test mean(marginalise(N1, K1)) ≈ Φ1 * μ
        @test cov(marginalise(N1, K1)) ≈ Hermitian(Φ1 * Σ * Φ1')

        @test mean(Nc) ≈ mean(N_gt)
        @test cov(Nc) ≈ cov(N_gt)
        @test cov(Kc)(x) ≈ cov(K_gt)(x)
        @test slope(mean(Kc)) ≈ slope(mean(K_gt))
        @test intercept(mean(Kc)) ≈ intercept(mean(K_gt))
    end
end
