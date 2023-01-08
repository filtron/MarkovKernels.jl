function marginalise_test(T, n, m, cov_types, matrix_types)
    μp = randn(T, n)
    Σp = randn(T, n, n)
    Σp = Σp * Σp'
    Ap = randn(T, n, n)
    Qp = randn(T, n, n)
    Qp = Qp * Qp'

    for cov_t in cov_types, matrix_t in matrix_types
        μ = _make_vector(μp, matrix_t)
        Σ = _make_matrix(Σp, matrix_t)
        A = _make_matrix(Ap, matrix_t)
        Q = _make_matrix(Qp, matrix_t)

        N = Normal(μ, _make_covp(Σ, cov_t))
        D = Dirac(μ)
        NK = NormalKernel(A, _make_covp(Q, cov_t))
        DK = DiracKernel(A)

        for distribution in (N, D), kernel in (NK, DK)
            _test_pair_marginalise(distribution, kernel, (μ, Σ), (A, Q))
        end
    end
end

function _test_pair_marginalise(D::Normal, K::AffineNormalKernel, P1, P2)
    μ, Σ = P1
    A, Q = P2
    @testset "marginalise | $(nameof(typeof(D))) | $(nameof(typeof(K)))" begin
        @test mean(marginalise(D, K)) ≈ A * μ
        @test cov(marginalise(D, K)) ≈ A * Σ * A' + Q
    end
end

function _test_pair_marginalise(D::Normal, K::AffineDiracKernel, P1, P2)
    μ, Σ = P1
    A = P2[1]
    @testset "marginalise | $(nameof(typeof(D))) | $(nameof(typeof(K)))" begin
        @test mean(marginalise(D, K)) ≈ A * μ
        @test cov(marginalise(D, K)) ≈ A * Σ * A'
    end
end

function _test_pair_marginalise(D::Dirac, K::AffineNormalKernel, P1, P2)
    μ, Σ = P1
    A, Q = P2
    @testset "marginalise | $(nameof(typeof(D))) | $(nameof(typeof(K)))" begin
        @test mean(marginalise(D, K)) ≈ A * μ
        @test cov(marginalise(D, K)) ≈ Q
    end
end

function _test_pair_marginalise(D::Dirac, K::AffineDiracKernel, P1, P2)
    μ, Σ = P1
    A, Q = P2
    @testset "marginalise | $(nameof(typeof(D))) | $(nameof(typeof(K)))" begin
        @test mean(marginalise(D, K)) ≈ A * μ
    end
end

function _test_marginalse_particle_system(T, n, m)
    k = 10

    X = [randn(T, n) for i in 1:k]
    logws = randn(real(T), k)
    P = ParticleSystem(logws, X)

    C = randn(T, m, n)
    K = DiracKernel(C)

    @testset "marginalise | $(typeof(P)) | $(typeof(K))" begin
        @test dim(marginalise(P, K)) == m
        @test logweights(marginalise(P, K)) == logweights(P)
        @test particles(marginalise(P, K)) ≈ [C * X[i] for i in eachindex(X)]
    end
end
