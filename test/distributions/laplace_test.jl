@safetestset "Laplace" begin
    using MarkovKernels, LinearAlgebra
    import LinearAlgebra: HermOrSym
    import RecursiveArrayTools: recursivecopy, recursivecopy!

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

            @test dim(L1) == 1
            @test cov(L1) ≈ v1
            @test var(L1) ≈ v1
            @test std(L1) ≈ abs(l1)

            @test residual(L1, x) ≈ (x - m1) / lsqrt(v1)
            @test typeof(residual(L1, x)) == typeof(x)
            @test logpdf(L1, x) ≈ -log(sqrt(2*v1)) - sqrt(2) * abs(residual(L1, x))

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
end
