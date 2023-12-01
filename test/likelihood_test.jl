function likelihood_test(T, n, m, cov_types, matrix_types)
    Cp = randn(T, n, m)
    xp = randn(T, m)
    yp = randn(T, n)
    Rp = randn(T, n, n)
    Rp = Rp * Rp'

    for cov_t in cov_types, matrix_t in matrix_types
        C = _make_matrix(Cp, matrix_t)
        x = _make_vector(xp, matrix_t)
        y = _make_vector(yp, matrix_t)
        R = _make_covp(_make_matrix(Rp, matrix_t), cov_t)

        K = NormalKernel(C, R)
        L = Likelihood(K, y)
        @testset "Likelihood | AffineNormal | {$(T),$(cov_t),$(matrix_t)}" begin
            @test L == Likelihood(K, y)
            @test measurement(L) == y
            @test measurement_model(L) == K
            @test log(L, x) ≈ logpdf(condition(K, x), y)
        end

        K = DiracKernel(C)
        L = Likelihood(K, y)
        @testset "Likelihood | AffineDirac | {$(T),$(cov_t),$(matrix_t)}" begin
            @test L == Likelihood(K, y)
            @test measurement(L) == y
            @test measurement_model(L) == K
        end

        @testset "FlatLikelihood" begin
            L1 = FlatLikelihood()
            L2 = FlatLikelihood()
            x = randn(T, 1)
            @test_nowarn FlatLikelihood()
            @test L1 === L2
            @test log(L1, x) == zero(real(eltype(x)))
            @test typeof(log(L1, x)) == typeof(zero(real(eltype(x))))
        end
    end
end
