function compose_test(T, n, cov_types, matrix_types)
    A1p = randn(T, n, n)
    A2p = randn(T, n, n)
    V1p = randn(T, n, n)
    V1p = V1p * V1p'
    V2p = randn(T, n, n)
    V2p = V2p * V2p'
    xp = randn(T, n)

    for cov_t in cov_types, matrix_t in matrix_types
        A1 = _make_matrix(A1p, matrix_t)
        A2 = _make_matrix(A2p, matrix_t)
        Σ1 = _make_matrix(V1p, matrix_t)
        Σ2 = _make_matrix(V2p, matrix_t)
        x = _make_vector(xp, matrix_t)

        F1 = LinearMap(A1)
        F2 = LinearMap(A2)
        NK1 = NormalKernel(F1, _make_covp(Σ1, cov_t))
        NK2 = NormalKernel(F2, _make_covp(Σ2, cov_t))

        DK1 = DiracKernel(F1)
        DK2 = DiracKernel(F2)

        IK = IdentityKernel()

        kernels = (NK1, NK2, DK1, DK2, IK)

        for K1 in kernels, K2 in kernels
            _test_pair_compose(K1, K2)
        end
    end
end

function _test_pair_compose(K2::AffineNormalKernel, K1::AffineNormalKernel)
    A2, b2, Σ2 = slope(mean(K2)), intercept(mean(K2)), Matrix(covp(K2))
    A1, b1, Σ1 = slope(mean(K1)), intercept(mean(K1)), Matrix(covp(K1))
    K3 = compose(K2, K1)
    @testset "compose | $(nameof(typeof(K2))) | $(nameof(typeof(K1)))" begin
        @test K3 == K2 ∘ K1
        @test slope(mean(K3)) ≈ A2 * A1
        @test intercept(mean(K3)) ≈ A2 * b1 + b2
        @test Matrix(covp(K3)) ≈ A2 * Σ1 * A2' + Σ2
    end
end

function _test_pair_compose(K2::AffineNormalKernel, K1::AffineDiracKernel)
    A2, b2, Σ2 = slope(mean(K2)), intercept(mean(K2)), Matrix(covp(K2))
    A1, b1 = slope(mean(K1)), intercept(mean(K1))
    K3 = compose(K2, K1)
    @testset "compose | $(nameof(typeof(K2))) | $(nameof(typeof(K1)))" begin
        @test K3 == K2 ∘ K1
        @test slope(mean(K3)) ≈ A2 * A1
        @test intercept(mean(K3)) ≈ A2 * b1 + b2
        @test Matrix(covp(K3)) ≈ Σ2
    end
end

function _test_pair_compose(K2::AffineDiracKernel, K1::AffineDiracKernel)
    A2, b2 = slope(mean(K2)), intercept(mean(K2))
    A1, b1 = slope(mean(K1)), intercept(mean(K1))
    K3 = compose(K2, K1)
    @testset "compose | $(nameof(typeof(K2))) | $(nameof(typeof(K1)))" begin
        @test K3 == K2 ∘ K1
        @test slope(mean(K3)) ≈ A2 * A1
        @test intercept(mean(K3)) ≈ A2 * b1 + b2
    end
end

function _test_pair_compose(K2::AffineDiracKernel, K1::AffineNormalKernel)
    A2, b2 = slope(mean(K2)), intercept(mean(K2))
    A1, b1, Σ1 = slope(mean(K1)), intercept(mean(K1)), Matrix(covp(K1))
    K3 = compose(K2, K1)
    @testset "compose | $(nameof(typeof(K2))) | $(nameof(typeof(K1)))" begin
        @test K3 == K2 ∘ K1
        @test slope(mean(K3)) ≈ A2 * A1
        @test intercept(mean(K3)) ≈ A2 * b1 + b2
        @test Matrix(covp(K3)) ≈ A2 * Σ1 * A2'
    end
end

function _test_pair_compose(K2::AbstractMarkovKernel, K1::IdentityKernel)
    K3 = compose(K2, K1)
    @testset "compose | $(nameof(typeof(K2))) | $(nameof(typeof(K1)))" begin
        @test K3 == K2 ∘ K1
        @test K3 == K2
    end
end

function _test_pair_compose(K2::IdentityKernel, K1::AbstractMarkovKernel)
    K3 = compose(K2, K1)
    @testset "compose | $(nameof(typeof(K2))) | $(nameof(typeof(K1)))" begin
        @test K3 == K2 ∘ K1
        @test K3 == K1
    end
end

function _test_pair_compose(K2::IdentityKernel, K1::IdentityKernel)
    K3 = compose(K2, K1)
    @testset "compose | $(nameof(typeof(K2))) | $(nameof(typeof(K1)))" begin
        @test K3 == K2 ∘ K1
        @test K3 == K1
    end
end
