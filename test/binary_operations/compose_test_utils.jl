function _test_pair_compose(
    K2::AffineHomoskedasticNormalKernel,
    K1::AffineHomoskedasticNormalKernel,
)
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

function _test_pair_compose(K2::AffineHomoskedasticNormalKernel, K1::AffineDiracKernel)
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

function _test_pair_compose(K2::AffineDiracKernel, K1::AffineHomoskedasticNormalKernel)
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
