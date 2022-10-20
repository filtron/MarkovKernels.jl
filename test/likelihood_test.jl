function likelihood_test(T, n, m)
    Φ1 = randn(T, n, m)
    RQ1 = randn(T, n, n)
    Q1 = RQ1' * RQ1
    K1 = NormalKernel(Φ1, Q1)

    μ = randn(T, m)
    RΣ = randn(T, m, m)
    Σ = RΣ' * RΣ

    N = Normal(μ, Σ)

    x = rand(N)
    y = rand(condition(K1, x))
    L = Likelihood(K1, y)

    S, G, Π = _schur(Σ, Φ1, Q1)
    pred = Φ1 * μ
    C = Normal(μ + G * (y - pred), Π)
    M = Normal(pred, S)
    Cgt1, loglike1 = bayes_rule(N, y, K1)
    Cgt2, loglike2 = bayes_rule(N, L)

    @testset "Normal Likelihood | $(T) " begin
        @test measurement(L) == y
        @test slope(mean(measurement_model(L))) == Φ1

        @test mean(C) ≈ mean(Cgt1) ≈ mean(Cgt2)
        @test cov(C) ≈ cov(Cgt1) ≈ cov(Cgt2)
        @test logpdf(M, y) ≈ loglike1 ≈ loglike2
    end
end
