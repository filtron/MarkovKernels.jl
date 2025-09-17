@safetestset "Laplace" begin
    using MarkovKernels, LinearAlgebra
    import LinearAlgebra: HermOrSym
    import RecursiveArrayTools: recursivecopy, recursivecopy!
    include("normal_test_utils.jl")
    n = 2
    etys = (Float64, Float32)
    affine_types = (LinearMap, AffineMap, AffineCorrector)
    cov_types = (HermOrSym, Cholesky)

    for T in etys
        @testset "UnivariateLaplace | $(T)" begin
            m1 = randn(T)
            l1 = randn(T)
            v1 = abs2(l1)
            x = randn(T)

            L1 = Laplace(m1, v1)

            @test_nowarn repr(L1)

            @test dim(L1) == 1
            @test cov(L1) ≈ v1
            @test var(L1) ≈ v1
            @test std(L1) ≈ abs(l1)

            @test residual(L1, x) ≈ (x - m1) / lsqrt(v1)
            @test typeof(residual(L1, x)) == typeof(x)
            @test isapprox(
                logpdf(L1, x),
                -log(sqrt(2*v1)) - sqrt(2) * abs(residual(L1, x)),
                atol = 1e-6,
            )

            @test eltype(var(L1)) <: Real
            @test eltype(std(L1)) <: Real
            @test eltype(logpdf(L1, x)) <: Real
            @test eltype(entropy(L1)) <: Real

            @test length(rand(L1)) == dim(L1)
            @test eltype(rand(L1)) == sample_eltype(L1)
            @test typeof(rand(L1)) == sample_type(L1)
            @test eltype(rand(L1)) == sample_eltype(L1)
        end
    end

    for cov_t in cov_types
        for T in etys
            x = randn(T, n)

            μ = randn(T, n)
            V1p = tril(ones(T, n, n))
            V1p = V1p * V1p'
            Σ = _make_covp(V1p, cov_t)

            L = Laplace(μ, Σ)

            @testset "Laplace | Unary | $(T) | " begin
                @test_nowarn repr(L)

                @test sample_type(L) == typeof(mean(L))

                @test !(copy(L) === L)
                @test copy(L) == L
                @test typeof(copy(L)) === typeof(L)
                @test typeof(similar(L)) === typeof(L)
                _L = similar(L)
                @test (copy!(_L, L); _L) == L

                @test L == L
                @test location(L) == mean(L) == μ
                @test cov(L) ≈ V1p
                @test covp(L) == Σ
                @test var(L) ≈ real.(diag(V1p))
                @test std(L) ≈ sqrt.(real.(diag(V1p)))

                @test residual(L, x) ≈ cholesky(V1p).L \ (x - μ)

                @test eltype(var(L)) <: Real
                @test eltype(std(L)) <: Real
                @test eltype(logpdf(L, x)) <: Real
                @test eltype(entropy(L)) <: Real

                @test length(rand(L)) == dim(L)
                @test eltype(rand(L)) == T
                @test typeof(rand(L)) == sample_type(L)
                @test eltype(rand(L)) == sample_eltype(L)
            end
        end
    end
end
