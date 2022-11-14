function invert_test(T, n, m, cov_types, matrix_types)
    μp = randn(T, m)
    Σp = randn(T, m, m)
    Σp = Σp * Σp'
    Ap = randn(T, n, m)
    Rp = randn(T, n, n)
    Rp = Rp * Rp'

    yp = randn(T, n)

    for cov_t in cov_types, matrix_t in matrix_types
        μ = _make_vector(μp, matrix_t)
        Σ = _make_matrix(Σp, matrix_t)
        A = _make_matrix(Ap, matrix_t)
        R = _make_matrix(Rp, matrix_t)
        y = _make_vector(yp, matrix_t)

        N = Normal(μ, _make_covp(Σ, cov_t))
        NK = NormalKernel(A, _make_covp(R, cov_t))
        DK = DiracKernel(A)

        S, G, Π = _schur(Σ, A, R)
        Ngt = Normal(A * μ, S)
        Kgt = NormalKernel(G, μ, A * μ, Π)

        NC, KC = invert(N, NK)

        @testset "invert | $(nameof(typeof(N))) | $(nameof(typeof(NK)))" begin
            @test mean(NC) ≈ mean(Ngt)
            @test cov(NC) ≈ cov(Ngt)

            @test slope(mean(KC)) ≈ slope(mean(Kgt))
            @test intercept(mean(KC)) ≈ intercept(mean(Kgt))
            @test cov(condition(KC, y)) ≈ cov(condition(Kgt, y))
            @test mean(condition(KC, y)) ≈ mean(condition(Kgt, y))
        end

        S, G, Π = _schur(Σ, A)
        Ngt = Normal(A * μ, S)
        Kgt = NormalKernel(G, μ, A * μ, Π)

        NC, KC = invert(N, DK)

        @testset "invert | $(nameof(typeof(N))) | $(nameof(typeof(DK)))" begin
            @test mean(NC) ≈ mean(Ngt)
            @test cov(NC) ≈ cov(Ngt)

            @test slope(mean(KC)) ≈ slope(mean(Kgt))
            @test intercept(mean(KC)) ≈ intercept(mean(Kgt))
            @test cov(condition(KC, y)) ≈ cov(condition(Kgt, y))
            @test mean(condition(KC, y)) ≈ mean(condition(Kgt, y))
        end
    end
end
