
function normal_test(T, n)
    μ1 = randn(T, n)
    L1 = randn(T, n, n)
    Σ1 = Hermitian(L1 * L1')

    x1 = randn(T, n)

    μ2 = randn(T, n)
    L2 = randn(T, n, n)
    Σ2 = Hermitian(L2 * L2')

    N1 = Normal(μ1, Σ1)
    N2 = Normal(μ2, Σ2)
    N12 = Normal(μ1, Σ1)

    λ1 = real(T)(2.0)
    λ2 = real(T)(3.0)
    IN1 = Normal(μ1, λ1 * I)
    IN2 = Normal(μ2, λ2 * I)

    @testset "Normal | $(T) " begin
        @test eltype(N1) == T
        @test typeof(similar(N1)) == typeof(N1)
        @test N1 == N12
        @test mean(N1) == μ1
        @test cov(N1) == Σ1
        @test var(N1) == diag(Σ1)
        @test std(N1) == sqrt.(diag(Σ1))

        @test residual(N1, x1) ≈ cholesky(Σ1).L \ (x1 - μ1)
        @test logpdf(N1, x1) ≈ _logpdf(T, μ1, Σ1, x1)

        @test entropy(N1) ≈ _entropy(T, μ1, Σ1)
        @test kldivergence(N1, N2) ≈ _kld(T, μ1, Σ1, μ2, Σ2)
        @test kldivergence(N2, N1) ≈ _kld(T, μ2, Σ2, μ1, Σ1)

        @test eltype(var(N1)) <: Real
        @test eltype(std(N1)) <: Real
        @test eltype(logpdf(N1, x1)) <: Real
        @test eltype(entropy(N1)) <: Real
        @test eltype(kldivergence(N1, N2)) <: Real
        @test eltype(kldivergence(N2, N1)) <: Real
    end

    @testset "IsoNormal | $(T) " begin
        @test eltype(IN1) == T
        @test IN1 == IsoNormal(μ1, λ1)
        @test mean(IN1) == μ1
        @test cov(IN1) == λ1 * I
        @test var(IN1) == fill(λ1, length(μ1))
        @test std(IN1) == sqrt.(fill(λ1, length(μ1)))

        @test residual(IN1, x1) ≈ (x1 - μ1) / sqrt(λ1)
        @test logpdf(IN1, x1) ≈ _logpdf(T, μ1, λ1 * I, x1)

        @test entropy(IN1) ≈ _entropy(T, μ1, λ1 * I)
        @test kldivergence(IN1, IN2) ≈ _kld(T, μ1, λ1 * I, μ2, λ2 * I)
        @test kldivergence(IN2, IN1) ≈ _kld(T, μ2, λ2 * I, μ1, λ1 * I)

        @test eltype(var(IN1)) <: Real
        @test eltype(std(IN1)) <: Real
        @test eltype(logpdf(IN1, x1)) <: Real
        @test eltype(entropy(IN1)) <: Real
        @test eltype(kldivergence(IN1, IN2)) <: Real
        @test eltype(kldivergence(IN2, IN1)) <: Real
    end

    @testset "Normal / IsoNormal | $(T) " begin
        @test kldivergence(N1, IN2) ≈ _kld(T, μ1, Σ1, μ2, λ2 * I)
        @test kldivergence(IN2, N1) ≈ _kld(T, μ2, λ2 * I, μ1, Σ1)
        @test eltype(kldivergence(N1, IN2)) <: Real
        @test eltype(kldivergence(IN2, N1)) <: Real
    end
end

function _logpdf(T, μ1, Σ1, x1)
    n = length(μ1)
    Σ1 = Hermitian(Σ1[1:n, 1:n])
    if T <: Real
        logpdf = -0.5 * logdet(2 * π * Σ1) - 0.5 * dot(x1 - μ1, inv(Σ1), x1 - μ1)
    elseif T <: Complex
        logpdf = -n * log(π) - logdet(Σ1) - dot(x1 - μ1, inv(Σ1), x1 - μ1)
    end

    return logpdf
end

function _entropy(T, μ1, Σ1)
    n = length(μ1)
    Σ1 = Hermitian(Σ1[1:n, 1:n])
    if T <: Real
        entropy = 1.0 / 2.0 * logdet(2 * π * exp(1) * Σ1)
    elseif T <: Complex
        entropy = n * log(π) + logdet(Σ1) + n
    end
end

function _kld(T, μ1, Σ1, μ2, Σ2)
    n = length(μ1)
    Σ1 = Hermitian(Σ1[1:n, 1:n])
    Σ2 = Hermitian(Σ2[1:n, 1:n])

    if T <: Real
        kld =
            1 / 2 *
            (tr(Σ2 \ Σ1) - n + dot(μ2 - μ1, inv(Σ2), μ2 - μ1) + logdet(Σ2) - logdet(Σ1))
    elseif T <: Complex
        kld =
            real(tr(Σ2 \ Σ1)) - n + real(dot(μ2 - μ1, inv(Σ2), μ2 - μ1)) + logdet(Σ2) -
            logdet(Σ1)
    end

    return kld
end
