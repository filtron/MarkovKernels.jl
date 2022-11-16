function mixture_test()
    μ1 = [1.0, 2.0]
    μ2 = [3.0, 1.0]

    Σ1 = [1.0 1.0; 1.0 2.0]
    Σ2 = diagm(ones(2))

    ws = [0.3, 0.7]
    M1 = Mixture(ws, [Normal(μ1, Σ1), Normal(μ2, Σ2)])
    M2 = Mixture(ws, [Dirac(μ1), Dirac(μ2)])

    m = [μ1 μ2] * ws
    covmean = [μ1 μ2] * diagm(ws) * [μ1 μ2]' - m * m'
    meancov = ws[1] * Σ1 + ws[2] * Σ2

    @testset "Mixture" begin
        @test mean(M1) ≈ mean(M2) ≈ m
        @test cov(M1) ≈ meancov + covmean
    end
end
