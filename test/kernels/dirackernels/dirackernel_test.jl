@testset "DiracKernels | DiracKernel" begin
    etys = (Float64, Complex{Float64})
    m, n = 2, 3
    for T in etys
        μ1(x) = [x[1]^2 - x[2]^2, x[2]^2 - x[3]^2]
        K1 = DiracKernel(μ1)
        x1 = randn(T, n)

        μ2(x) = sin(x)
        K2 = DiracKernel(μ2)
        x2 = randn(T)

        @testset "DiracKernels | DiracKernel | Unary | $(T)" begin
            @test_nowarn repr(K1)
            @test mean(K1)(x1) == μ1(x1)
            @test condition(K1, x1) == Dirac(μ1(x1))

            @test mean(K2)(x2) == μ2(x2)
            @test condition(K2, x2) == Dirac(μ2(x2))
        end
    end

    for T in etys
        # vector to vector
        C1 = randn(T, m, n)
        x1 = randn(T, n)
        F1 = LinearMap(C1)
        K1 = DiracKernel(F1)

        @testset "DiracKernels | AffineDiracKernel | Unary | $(T)" begin
            @test_nowarn repr(K1)
            @test !(copy(K1) === K1)
            @test typeof(copy(K1)) === typeof(K1)
            @test typeof(similar(K1)) === typeof(K1)
            @test copy!(similar(K1), K1) == K1

            @test typeof(K1) <: AffineDiracKernel
            @test mean(K1)(x1) == F1(x1)
            @test condition(K1, x1) == Dirac(F1(x1))
        end

        # vector to scalar
        C2 = adjoint(randn(T, n))
        x2 = randn(T, n)
        F2 = LinearMap(C2)
        K2 = DiracKernel(F2)

        # scalar to scalar
        C3 = randn(T)
        x3 = randn(T)
        F3 = LinearMap(C3)
        K3 = DiracKernel(F3)

        @testset "DiracKernels | AffineDiracKernel | Unary | $(T)" begin
            @test typeof(K2) <: AffineDiracKernel
            @test mean(K2)(x2) == F2(x2)
            @test condition(K2, x2) == Dirac(F2(x2))

            @test typeof(K3) <: AffineDiracKernel
            @test mean(K3)(x3) == F3(x3)
            @test condition(K3, x3) == Dirac(F3(x3))
        end
    end
end
