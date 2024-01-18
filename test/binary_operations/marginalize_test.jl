function marginalize_test(T, n, m, cov_types, matrix_types)
    μp = randn(T, n)
    Σp = randn(T, n, n)
    Σp = Σp * Σp'
    Ap = randn(T, n, n)
    Qp = randn(T, n, n)
    Qp = Qp * Qp'

    vp = randn(T, n)
    Cp = randn(T, n, n)

    for cov_t in cov_types, matrix_t in matrix_types
        μ = _make_vector(μp, matrix_t)
        Σ = _make_matrix(Σp, matrix_t)
        A = _make_matrix(Ap, matrix_t)
        Q = _make_matrix(Qp, matrix_t)

        N = Normal(μ, _make_covp(Σ, cov_t))
        D = Dirac(μ)
        NK = NormalKernel(A, _make_covp(Q, cov_t))
        DK = DiracKernel(A)
        IK = IdentityKernel()

        # marginalize 
        for distribution in (N, D), kernel in (NK, DK, IK)
            _test_pair_marginalize(distribution, kernel)
        end

        # plus / minus  
        v = _make_vector(vp, matrix_t)
        for distribution in (N, D)
            @test mean(distribution + v) == mean(v + distribution) == mean(distribution) + v
            @test mean(distribution - v) == mean(distribution) - v
            @test mean(v - distribution) == v - mean(distribution)
        end

        # multiplication 
        C = _make_matrix(Cp, matrix_t)
        for distribution in (N, D)
            @test C * distribution == marginalize(distribution, DiracKernel(C))
        end
    end
end

function _test_pair_marginalize(D::Normal, K::AffineNormalKernel)
    μ, Σ = mean(D), Matrix(covp(D))
    A, Q = slope(mean(K)), Matrix(covp(K))
    @testset "marginalize | $(nameof(typeof(D))) | $(nameof(typeof(K)))" begin
        @test mean(marginalize(D, K)) ≈ A * μ
        @test cov(marginalize(D, K)) ≈ A * Σ * A' + Q
    end
end

function _test_pair_marginalize(D::Normal, K::AffineDiracKernel)
    μ, Σ = mean(D), Matrix(covp(D))
    A = slope(mean(K))
    @testset "marginalize | $(nameof(typeof(D))) | $(nameof(typeof(K)))" begin
        @test mean(marginalize(D, K)) ≈ A * μ
        @test cov(marginalize(D, K)) ≈ A * Σ * A'
    end
end

function _test_pair_marginalize(D::Dirac, K::AffineNormalKernel)
    μ = mean(D)
    A, Q = slope(mean(K)), Matrix(covp(K))
    @testset "marginalize | $(nameof(typeof(D))) | $(nameof(typeof(K)))" begin
        @test mean(marginalize(D, K)) ≈ A * μ
        @test cov(marginalize(D, K)) ≈ Q
    end
end

function _test_pair_marginalize(D::Dirac, K::AffineDiracKernel)
    μ = mean(D)
    A, b = slope(mean(K)), intercept(mean(K))
    @testset "marginalize | $(nameof(typeof(D))) | $(nameof(typeof(K)))" begin
        @test mean(marginalize(D, K)) ≈ A * μ + b
    end
end

function _test_pair_marginalize(D::AbstractDistribution, K::IdentityKernel)
    @testset "marginalize | $(nameof(typeof(D))) | $(nameof(typeof(K)))" begin
        @test D === marginalize(D, K)
    end
end

function _test_marginalze_particle_system(T, n, m)
    k = 10

    X = [randn(T, n) for i in 1:k]
    logws = randn(real(T), k)
    P = ParticleSystem(logws, X)

    C = randn(T, m, n)
    K = DiracKernel(C)

    @testset "marginalize | $(typeof(P)) | $(typeof(K))" begin
        @test dim(marginalize(P, K)) == m
        @test logweights(marginalize(P, K)) == logweights(P)
        @test particles(marginalize(P, K)) ≈ [C * X[i] for i in eachindex(X)]
    end
end
