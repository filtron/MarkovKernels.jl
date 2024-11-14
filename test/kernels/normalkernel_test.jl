@testset "NormalKernel" begin
    etys = (Float64, ComplexF64)
    m, n = 2, 3

    @testset "NormalKernel | NonlinearNormalKernel" begin
        for T in etys

            # MIMO
            μ1(x) = [exp(x[1]), sin(x[2])]
            Σ1(x) = Cholesky(UpperTriangular([exp(x[1]) one(T); zero(T) abs(x[2])]))
            x1 = randn(T, 2)
            K1 = NormalKernel(μ1, Σ1)

            # MISO
            μ2(x) = x[1] - sin(x[2])
            Σ2(x) = norm(x)
            x2 = randn(T, 2)
            K2 = NormalKernel(μ2, Σ2)

            # SISO
            μ3(x) = sin(x)
            Σ3(x) = abs(x)
            x3 = randn(T)
            K3 = NormalKernel(μ3, Σ3)

            μs = [μ1, μ2, μ3]
            Σs = [Σ1, Σ2, Σ3]
            Ks = [K1, K2, K3]
            xs = [x1, x2, x3]

            for (μ, Σ, K, x) in zip(μs, Σs, Ks, xs)
                @test_nowarn repr(K)
                @test typeof(K) <: NonlinearNormalKernel
                @test mean(K)(x) == μ(x)
                @test cov(K)(x) == Σ(x)
                @test condition(K, x) == Normal(μ(x), Σ(x))

                @test typeof_sample(condition(K, x)) == typeof(rand(K, x))
                @test eltype_sample(condition(K, x)) == eltype(rand(K, x))
            end
        end
    end

    @testset "NormalKernel | AffineHeteroskedasticNormalKernel" begin
        for T in etys

            # MIMO
            A1 = randn(T, m, n)
            b1 = randn(T, m)
            x1 = randn(T, n)
            μ1 = AffineMap(A1, b1)
            Σ1(x) = Cholesky(UpperTriangular([exp(x[1]) one(T); zero(T) abs(x[2])]))
            K1 = NormalKernel(μ1, Σ1)

            # MISO
            A2 = adjoint(randn(T, m))
            b2 = randn(T)
            x2 = randn(T, m)
            μ2 = AffineMap(A2, b2)
            Σ2(x) = norm(x)
            K2 = NormalKernel(μ2, Σ2)

            # SISO
            A3 = randn(T)
            b3 = randn(T)
            x3 = randn(T)
            μ3 = AffineMap(A3, b3)
            Σ3(x) = abs(x)
            K3 = NormalKernel(μ3, Σ3)

            μs = [μ1, μ2, μ3]
            Σs = [Σ1, Σ2, Σ3]
            Ks = [K1, K2, K3]
            xs = [x1, x2, x3]

            for (μ, Σ, K, x) in zip(μs, Σs, Ks, xs)
                @test_nowarn repr(K)
                @test typeof(K) <: AffineHeteroskedasticNormalKernel
                @test mean(K)(x) == μ(x)
                @test cov(K)(x) == Σ(x)
                @test condition(K, x) == Normal(μ(x), Σ(x))
            end
        end
    end

    @testset "NormalKernel | HomoskedasticNormalKernel" begin
        for T in etys

            # MIMO
            μ1(x) = [exp(x[1]), sin(x[2])]
            Σ1 = Cholesky(UpperTriangular(ones(T, 2, 2)))
            x1 = randn(T, 2)
            K1 = NormalKernel(μ1, Σ1)

            # MISO
            μ2(x) = x[1] - sin(x[2])
            Σ2 = one(real(T))
            x2 = randn(T, 2)
            K2 = NormalKernel(μ2, Σ2)

            # SISO
            μ3(x) = sin(x)
            Σ3 = one(real(T))
            x3 = randn(T)
            K3 = NormalKernel(μ3, Σ3)

            μs = [μ1, μ2, μ3]
            Σs = [Σ1, Σ2, Σ3]
            Ks = [K1, K2, K3]
            xs = [x1, x2, x3]
            for (μ, Σ, K, x) in zip(μs, Σs, Ks, xs)
                @test_nowarn repr(K)
                @test typeof(K) <: HomoskedasticNormalKernel
                @test mean(K)(x) == μ(x)
                @test cov(K)(x) == Σ
                @test condition(K, x) == Normal(μ(x), Σ)
                @test typeof_sample(condition(K, x)) == typeof(rand(K, x))
                @test eltype_sample(condition(K, x)) == eltype(rand(K, x))
            end
        end
    end

    @testset "NormalKernel | AffineHomoskedasticNormalKernel" begin
        for T in etys
            # MIMO
            A1 = randn(T, m, n)
            b1 = randn(T, m)
            x1 = randn(T, n)
            μ1 = AffineMap(A1, b1)
            Σ1 = Cholesky(UpperTriangular(ones(T, 2, 2)))
            K1 = NormalKernel(μ1, Σ1)

            # MISO
            A2 = adjoint(randn(T, m))
            b2 = randn(T)
            x2 = randn(T, m)
            μ2 = AffineMap(A2, b2)
            Σ2 = one(real(T))
            K2 = NormalKernel(μ2, Σ2)

            # SISO
            A3 = randn(T)
            b3 = randn(T)
            x3 = randn(T)
            μ3 = AffineMap(A3, b3)
            Σ3 = one(real(T))
            K3 = NormalKernel(μ3, Σ3)

            μs = [μ1, μ2, μ3]
            Σs = [Σ1, Σ2, Σ3]
            Ks = [K1, K2, K3]
            xs = [x1, x2, x3]
            for (μ, Σ, K, x) in zip(μs, Σs, Ks, xs)
                !isbits(K) && @test !(copy(K) === K)
                @test typeof(copy(K)) === typeof(K)
                #  @test_broken typeof(similar(K)) === typeof(K)
                #  @test_broken copy!(similar(K), K) == K

                @test_nowarn repr(K)
                @test typeof(K) <: AffineHomoskedasticNormalKernel
                @test mean(K)(x) == μ(x)
                @test cov(K)(x) == Σ
                @test condition(K, x) == Normal(μ(x), Σ)
                @test typeof_sample(condition(K, x)) == typeof(rand(K, x))
                @test eltype_sample(condition(K, x)) == eltype(rand(K, x))
            end
        end
    end
end
