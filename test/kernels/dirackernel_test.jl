@safetestset "DiracKernel" begin
    using MarkovKernels, LinearAlgebra
    etys = (Float64, ComplexF64)
    m, n = 2, 3

    @testset "DiracKernel | NonlinearDiracKernel" begin
        for T in etys
            # MIMO
            μ1(x) = [exp(x[1]), sin(x[2])]
            x1 = randn(T, 2)
            K1 = DiracKernel(μ1)

            # MISO
            μ2(x) = x[1] - sin(x[2])
            x2 = randn(T, 2)
            K2 = DiracKernel(μ2)

            # SISO
            μ3(x) = sin(x)
            x3 = randn(T)
            K3 = DiracKernel(μ3)

            μs = [μ1, μ2, μ3]
            Ks = [K1, K2, K3]
            xs = [x1, x2, x3]

            for (μ, K, x) in zip(μs, Ks, xs)
                @test_nowarn repr(K)
                @test typeof(K) <: DiracKernel

                @test mean(K)(x) == μ(x)
                @test condition(K, x) == Dirac(μ(x))
                @test sample_type(condition(K, x)) == typeof(rand(K, x))
                @test sample_eltype(condition(K, x)) == eltype(rand(K, x))
            end
        end
    end

    @testset "DiracKernel | AffineDiracKernel" begin
        for T in etys
            # MIMO
            A1 = randn(T, m, n)
            b1 = randn(T, m)
            x1 = randn(T, n)
            μ1 = AffineMap(A1, b1)
            K1 = DiracKernel(μ1)

            # MISO
            A2 = adjoint(randn(T, m))
            b2 = randn(T)
            x2 = randn(T, m)
            μ2 = AffineMap(A2, b2)
            K2 = DiracKernel(μ2)

            # SISO
            A3 = randn(T)
            b3 = randn(T)
            x3 = randn(T)
            μ3 = AffineMap(A3, b3)
            K3 = DiracKernel(μ3)

            μs = [μ1, μ2, μ3]
            Ks = [K1, K2, K3]
            xs = [x1, x2, x3]
            for (μ, K, x) in zip(μs, Ks, xs)
                !isbits(K) && @test !(copy(K) === K)
                @test typeof(copy(K)) === typeof(K)
                #  @test_broken typeof(similar(K)) === typeof(K)
                #  @test_broken copy!(similar(K), K) == K

                @test_nowarn repr(K)
                @test typeof(K) <: AffineDiracKernel
                @test mean(K)(x) == μ(x)
                @test condition(K, x) == Dirac(μ(x))
                @test sample_type(condition(K, x)) == typeof(rand(K, x))
                @test sample_eltype(condition(K, x)) == eltype(rand(K, x))
            end
        end
    end
end
