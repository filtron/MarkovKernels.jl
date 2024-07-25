function _logpdf(T, μ1, Σ1, x1)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1)
    if T <: Real
        logpdf = -T(0.5) * logdet(T(2π) * Σ1) - T(0.5) * dot(x1 - μ1, inv(Σ1), x1 - μ1)
    elseif T <: Complex
        logpdf = -real(T)(n) * log(real(T)(π)) - logdet(Σ1) - dot(x1 - μ1, inv(Σ1), x1 - μ1)
    end

    return logpdf
end

function _entropy(T, μ1, Σ1)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1)
    if T <: Real
        entropy = T(0.5) * logdet(T(2π) * exp(T(1)) * Σ1)
    elseif T <: Complex
        entropy = real(T)(n) * log(real(T)(π)) + logdet(Σ1) + real(T)(n)
    end
end

function _kld(T, μ1, Σ1, μ2, Σ2)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1)
    Σ2 = _symmetrise(T, Σ2)

    if T <: Real
        kld =
            T(0.5) *
            (tr(Σ2 \ Σ1) - T(n) + dot(μ2 - μ1, inv(Σ2), μ2 - μ1) + logdet(Σ2) - logdet(Σ1))
    elseif T <: Complex
        kld =
            real(tr(Σ2 \ Σ1)) - real(T)(n) +
            real(dot(μ2 - μ1, inv(Σ2), μ2 - μ1)) +
            logdet(Σ2) - logdet(Σ1)
    end

    return kld
end

@testset "Normal" begin
    etys = (Float64, Complex{Float64})
    matrix_types = (Matrix,)
    affine_types = (LinearMap, AffineMap, AffineCorrector)
    cov_types = (HermOrSym, Cholesky)

    for T in etys
        eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

        xp = randn(T, n)
        μ1p = randn(T, n)
        V1p = tril(ones(T, n, n))
        V1p = V1p * V1p'

        @testset "UvNormal | $(T)" begin
            m1 = randn(T)
            l1 = randn(T)
            v1 = abs2(l1)
            uvN1 = Normal(m1, v1)

            m2 = randn(T)
            l2 = randn(T)
            v2 = abs2(l2)
            uvN2 = Normal(m2, v2)

            x = randn(T)

            @test cov(uvN1) ≈ v1
            @test var(uvN1) ≈ v1
            @test std(uvN1) ≈ abs(l1)

            # add tests for Normal{T}(::UvNormal)
            for T2 in (Float32, Float64, ComplexF32, ComplexF64)
                if T <: Complex && T2 <: Real
                    @test_throws InexactError AbstractDistribution{T2}(uvN1)
                    @test_throws InexactError AbstractNormal{T2}(uvN1)
                    @test_throws InexactError Normal{T2}(uvN1)
                else
                    @test AbstractDistribution{T2}(uvN1) ==
                          AbstractNormal{T2}(uvN1) ==
                          Normal{T2}(uvN1)
                    @test eltype(AbstractNormal{T2}(uvN1)) == T2
                end
            end

            @test residual(uvN1, x) ≈ (x - m1) / lsqrt(v1)
            @test typeof(residual(uvN1, x)) == typeof(x)
            @test logpdf(uvN1, x) ≈ _logpdf(T, m1, v1, x)
            @test entropy(uvN1) ≈ _entropy(T, m1, v1)

            @test eltype(var(uvN1)) <: Real
            @test eltype(std(uvN1)) <: Real
            @test eltype(logpdf(uvN1, x)) <: Real
            @test eltype(entropy(uvN1)) <: Real

            @test length(rand(uvN1)) == dim(uvN1)
            @test eltype(rand(uvN1)) == T
            @test typeof(rand(uvN1)) == typeof_sample(uvN1)
            @test eltype(rand(uvN1)) == eltype_sample(uvN1)

            @test kldivergence(uvN1, uvN2) ≈ _kld(T, m1, v1, m2, v2)
            @test kldivergence(uvN2, uvN1) ≈ _kld(T, m2, v2, m1, v1)
            @test eltype(kldivergence(uvN1, uvN2)) <: Real
            @test eltype(kldivergence(uvN2, uvN1)) <: Real
        end

        for cov_t in cov_types, matrix_t in matrix_types
            x = _make_vector(xp, matrix_t)
            μ = _make_vector(μ1p, matrix_t)
            Σ = _make_covp(_make_matrix(V1p, matrix_t), cov_t)

            N = Normal(μ, Σ)

            @testset "Normal | Unary | $(T) | " begin
                @test_nowarn repr(N)
                @test eltype(N) == T

                @test !(copy(N) === N)
                @test copy(N) == N
                @test typeof(copy(N)) === typeof(N)
                @test typeof(similar(N)) === typeof(N)
                _N = similar(N)
                @test (copy!(_N, N); _N) == N

                if Σ isa AbstractMatrix
                    @test !(recursivecopy(N) === N)
                    @test recursivecopy(N) == N
                    @test typeof(recursivecopy(N)) === typeof(N)
                    _N = similar(N)
                    @test (recursivecopy!(_N, N); _N) == N
                end

                for U in eltypes
                    @test AbstractDistribution{U}(N) == AbstractNormal{U}(N) == Normal{U}(N)
                    @test eltype(AbstractNormal{U}(N)) == U
                end
                @test N == N
                @test mean(N) == μ
                @test cov(N) ≈ V1p
                @test covp(N) == Σ
                @test var(N) ≈ real.(diag(V1p))
                @test std(N) ≈ sqrt.(real.(diag(V1p)))

                @test residual(N, x) ≈ cholesky(V1p).L \ (x - μ)
                @test _ofsametype(x, residual(N, x))
                @test logpdf(N, x) ≈ _logpdf(T, μ, V1p, x)
                @test entropy(N) ≈ _entropy(T, μ, V1p)

                @test eltype(var(N)) <: Real
                @test eltype(std(N)) <: Real
                @test eltype(logpdf(N, x)) <: Real
                @test eltype(entropy(N)) <: Real

                @test length(rand(N)) == dim(N)
                @test eltype(rand(N)) == T
                @test typeof(rand(N)) == typeof_sample(N)
                @test eltype(rand(N)) == eltype_sample(N)
            end
        end

        μ2p = randn(T, n)
        V2p = triu(ones(T, n, n))
        V2p = V2p * V2p'

        for cov_t in cov_types, matrix_t in matrix_types
            x = _make_vector(xp, matrix_t)
            μ1 = _make_vector(μ1p, matrix_t)
            Σ1 = _make_covp(_make_matrix(V1p, matrix_t), cov_t)
            N1 = Normal(μ1, Σ1)
            μ2 = _make_vector(μ2p, matrix_t)
            Σ2 = _make_covp(_make_matrix(V2p, matrix_t), cov_t)
            N2 = Normal(μ2, Σ2)

            @testset "Normal | Binary | {$(T),$(cov_t),$(matrix_t)}" begin
                @test kldivergence(N1, N2) ≈ _kld(T, μ1p, V1p, μ2p, V2p)
                @test kldivergence(N2, N1) ≈ _kld(T, μ2p, V2p, μ1p, V1p)
                @test eltype(kldivergence(N1, N2)) <: Real
                @test eltype(kldivergence(N2, N1)) <: Real
            end
        end
    end
end
