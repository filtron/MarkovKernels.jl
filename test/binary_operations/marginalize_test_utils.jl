_mat_or_num(V) = Matrix(V)
_mat_or_num(V::Number) = V

function _test_pair_marginalize(D::Normal, K::AffineHomoskedasticNormalKernel)
    μ, Σ = mean(D), _mat_or_num(covp(D))
    A, Q = slope(mean(K)), _mat_or_num(covp(K))
    @testset "marginalize | $(nameof(typeof(D))) | $(nameof(typeof(K)))" begin
        @test mean(marginalize(D, K)) ≈ A * μ
        @test cov(marginalize(D, K)) ≈ A * Σ * A' + Q
    end
end

function _test_pair_marginalize(D::Normal, K::AffineDiracKernel)
    μ, Σ = mean(D), _mat_or_num(covp(D))
    A = slope(mean(K))
    @testset "marginalize | $(nameof(typeof(D))) | $(nameof(typeof(K)))" begin
        @test mean(marginalize(D, K)) ≈ A * μ
        @test cov(marginalize(D, K)) ≈ A * Σ * A'
    end
end

function _test_pair_marginalize(D::Dirac, K::AffineHomoskedasticNormalKernel)
    μ = mean(D)
    A, Q = slope(mean(K)), _mat_or_num(covp(K))
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
