_mat_or_num(V) = Matrix(V)
_mat_or_num(V::Number) = V

function _test_pair_forward_operator(K::AffineHomoskedasticNormalKernel, D::Normal)
    μ, Σ = mean(D), _mat_or_num(covp(D))
    A, Q = slope(mean(K)), _mat_or_num(covp(K))
    @testset "forward_operator | $(nameof(typeof(K))) | $(nameof(typeof(D)))" begin
        @test mean(forward_operator(K, D)) ≈ A * μ
        @test cov(forward_operator(K, D)) ≈ A * Σ * A' + Q
    end
end

function _test_pair_forward_operator(K::AffineDiracKernel, D::Normal)
    μ, Σ = mean(D), _mat_or_num(covp(D))
    A = slope(mean(K))
    @testset "forward_operator | $(nameof(typeof(K))) | $(nameof(typeof(D)))" begin
        @test mean(forward_operator(K, D)) ≈ A * μ
        @test cov(forward_operator(K, D)) ≈ A * Σ * A'
    end
end

function _test_pair_forward_operator(K::AffineHomoskedasticNormalKernel, D::Dirac)
    μ = mean(D)
    A, Q = slope(mean(K)), _mat_or_num(covp(K))
    @testset "forward_operator | $(nameof(typeof(K))) | $(nameof(typeof(D)))" begin
        @test mean(forward_operator(K, D)) ≈ A * μ
        @test cov(forward_operator(K, D)) ≈ Q
    end
end

function _test_pair_forward_operator(K::AffineDiracKernel, D::Dirac)
    μ = mean(D)
    A, b = slope(mean(K)), intercept(mean(K))
    @testset "forward_operator | $(nameof(typeof(K))) | $(nameof(typeof(D)))" begin
        @test mean(forward_operator(K, D)) ≈ A * μ + b
    end
end

function _test_pair_forward_operator(K::IdentityKernel, D::AbstractDistribution)
    @testset "forward_operator | $(nameof(typeof(K))) | $(nameof(typeof(D)))" begin
        @test D === forward_operator(K, D)
    end
end
