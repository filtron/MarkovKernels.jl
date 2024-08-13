@testset "NormalKernels | HomoskedasticNormalKernel" begin
    etys = (Float64, Complex{Float64})
    ctys = (SelfAdjoint, Cholesky)
    m, n = 2, 3

    for T in etys, CT in ctys
        μ1(x) = [x[1]^2 - x[2]^2, x[2]^2 - x[3]^2]
        V1 = randn(T, m, m)
        V1 = V1 * V1'
        Σ1 = _make_covp(V1, CT)
        K1 = HomoskedasticNormalKernel(μ1, Σ1)
        x1 = randn(T, n)

        μ2(x) = sin(x)
        LΣ2 = randn(real(T))
        Σ2 = abs2(LΣ2)
        K2 = HomoskedasticNormalKernel(μ2, Σ2)
        x2 = randn(T)

        @testset "NormalKernels | HomoskedasticNormalKernel | Unary | $(T) | $(CT)" begin
            @test_nowarn repr(K1)
            @test mean(K1)(x1) == μ1(x1)
            @test cov(K1)(x1) == Σ1
            @test condition(K1, x1) == Normal(μ1(x1), Σ1)

            @test mean(K2)(x2) == μ2(x2)
            @test cov(K2)(x2) == Σ2
            @test condition(K2, x2) == Normal(μ2(x2), Σ2)
        end
    end

    for T in etys
        for CT in ctys

            # vector to vector
            C1 = randn(T, m, n)
            V1 = randn(T, m, m)
            V1 = V1 * V1'
            Σ1 = _make_covp(V1, CT)
            x1 = randn(T, n)
            F1 = LinearMap(C1)
            K1 = HomoskedasticNormalKernel(F1, Σ1)

            @testset "NormalKernels | AffineHomoskedasticNormalKernel  | Unary | $(T) | $(CT)" begin
                @test_nowarn repr(K1)
                @test !(copy(K1) === K1)
                @test typeof(copy(K1)) === typeof(K1)
                @test typeof(similar(K1)) === typeof(K1)
                @test copy!(similar(K1), K1) == K1

                @test typeof(K1) <: AffineHomoskedasticNormalKernel
                @test mean(K1)(x1) == F1(x1)
                @test cov(K1)(x1) == Σ1
                @test covp(K1) == Σ1
                @test condition(K1, x1) == Normal(F1(x1), Σ1)
            end
        end

        # vector to scalar
        C2 = adjoint(randn(T, n))
        LΣ2 = randn(real(T))
        Σ2 = abs2(LΣ2)
        x2 = randn(T, n)
        F2 = LinearMap(C2)
        K2 = HomoskedasticNormalKernel(F2, Σ2)

        # scalar to scalar
        C3 = randn(T)
        LΣ3 = randn(real(T))
        Σ3 = abs2(LΣ3)
        x3 = randn(T)
        F3 = LinearMap(C3)
        K3 = HomoskedasticNormalKernel(F3, Σ3)

        @testset "NormalKernels | AffineHomoskedasticNormalKernel  | Unary | $(T) " begin
            @test typeof(K2) <: AffineHomoskedasticNormalKernel
            @test mean(K2)(x2) == F2(x2)
            @test cov(K2)(x2) == Σ2
            @test covp(K2) == Σ2
            @test condition(K2, x2) == Normal(F2(x2), Σ2)

            @test typeof(K3) <: AffineHomoskedasticNormalKernel
            @test mean(K3)(x3) == F3(x3)
            @test cov(K3)(x3) == Σ3
            @test covp(K3) == Σ3
            @test condition(K3, x3) == Normal(F3(x3), Σ3)
        end
    end
end
