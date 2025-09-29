@safetestset "Normal" begin
    using MarkovKernels, LinearAlgebra
    import LinearAlgebra: HermOrSym
    import RecursiveArrayTools: recursivecopy, recursivecopy!
    include("normal_test_utils.jl")
    n = 2
    etys = (Float64, ComplexF64)
    affine_types = (LinearMap, AffineMap, AffineCorrector)
    cov_types = (HermOrSym, Cholesky)

    for T in etys
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
            @test typeof(rand(uvN1)) == sample_type(uvN1)
            @test eltype(rand(uvN1)) == sample_eltype(uvN1)

            @test kldivergence(uvN1, uvN2) ≈ _kld(T, m1, v1, m2, v2)
            @test kldivergence(uvN2, uvN1) ≈ _kld(T, m2, v2, m1, v1)
            @test eltype(kldivergence(uvN1, uvN2)) <: Real
            @test eltype(kldivergence(uvN2, uvN1)) <: Real
        end

        for cov_t in cov_types
            x = randn(T, n)

            μ = randn(T, n)
            V1p = tril(ones(T, n, n))
            V1p = V1p * V1p'
            Σ = _make_covp(V1p, cov_t)

            N = Normal(μ, Σ)

            @testset "Normal | Unary | $(T) | " begin
                @test_nowarn repr(N)

                @test sample_type(N) == typeof(mean(N))

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

                @test N == N
                @test mean(N) == μ
                @test cov(N) ≈ V1p
                @test covparam(N) == Σ
                @test var(N) ≈ real.(diag(V1p))
                @test std(N) ≈ sqrt.(real.(diag(V1p)))

                @test residual(N, x) ≈ cholesky(V1p).L \ (x - μ)
                @test logpdf(N, x) ≈ _logpdf(T, μ, V1p, x)
                @test entropy(N) ≈ _entropy(T, μ, V1p)

                @test eltype(var(N)) <: Real
                @test eltype(std(N)) <: Real
                @test eltype(logpdf(N, x)) <: Real
                @test eltype(entropy(N)) <: Real

                @test length(rand(N)) == dim(N)
                @test eltype(rand(N)) == T
                @test typeof(rand(N)) == sample_type(N)
                @test eltype(rand(N)) == sample_eltype(N)
            end
        end

        for cov_t in cov_types
            x = randn(T, n)

            μ1 = randn(T, n)
            V1p = tril(ones(T, n, n))
            V1p = V1p * V1p'
            Σ1 = _make_covp(V1p, cov_t)
            N1 = Normal(μ1, Σ1)

            μ2 = randn(T, n)
            V2p = triu(ones(T, n, n))
            V2p = V2p * V2p'
            Σ2 = _make_covp(V2p, cov_t)
            N2 = Normal(μ2, Σ2)

            @testset "Normal | Binary | {$(T),$(cov_t)}" begin
                @test kldivergence(N1, N2) ≈ _kld(T, μ1, V1p, μ2, V2p)
                @test kldivergence(N2, N1) ≈ _kld(T, μ2, V2p, μ1, V1p)
                @test eltype(kldivergence(N1, N2)) <: Real
                @test eltype(kldivergence(N2, N1)) <: Real
            end
        end
    end
end
