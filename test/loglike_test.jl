function loglike_test(T, n, m, cov_types, matrix_types)
    μp = randn(T, m)
    Vp = randn(T, m, m)
    Vp = Vp * Vp'
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

        μ = _make_vector(μp, matrix_t)
        Σ = _make_covp(_make_matrix(Vp, matrix_t), cov_t)
        N = Normal(μ, Σ)

        K = NormalKernel(C, R)
        L = LogLike(K, y)
        @testset "LogLike | AffineNormal | {$(T),$(cov_t),$(matrix_t)}" begin
            @test L == LogLike(K, y)
            @test measurement(L) == y
            @test measurement_model(L) == K
            @test L(x) ≈ logpdf(condition(K, x), y)
        end

        @testset "Loglike | AffineNormal | bayes_rule | {$(T),$(cov_t),$(matrix_t)}" begin
            M, KC = invert(N, K)

            NC, loglike = bayes_rule(N, L)
            @test mean(NC) ≈ mean(condition(KC, y))
            @test cov(NC) ≈ cov(condition(KC, y))
            @test loglike ≈ logpdf(M, y)
        end

        K = DiracKernel(C)
        L = LogLike(K, y)
        @testset "LogLike | AffineDirac | {$(T),$(cov_t),$(matrix_t)}" begin
            @test L == LogLike(K, y)
            @test measurement(L) == y
            @test measurement_model(L) == K
        end

        @testset "Loglike | AffineDirac | bayes_rule | {$(T),$(cov_t),$(matrix_t)}" begin
            M, KC = invert(N, K)

            NC, loglike = bayes_rule(N, L)
            @test mean(NC) ≈ mean(condition(KC, y))
            @test cov(NC) ≈ cov(condition(KC, y))
            @test loglike ≈ logpdf(M, y)
        end
    end
end
