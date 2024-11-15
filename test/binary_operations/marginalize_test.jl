@testset "marginalize" begin
    include("marginalize_test_utils.jl")

    m = 2
    etys = (Float64, ComplexF64)

    for T in etys
        @testset "marginalize | multivariate" begin
            μ1 = randn(T, m)
            V1 = Cholesky(UpperTriangular(ones(T, m, m)))

            N = Normal(μ1, V1)
            D = Dirac(μ1)

            # MIMO
            μ2 = LinearMap(randn(T, m, m))
            V2 = Cholesky(UpperTriangular(ones(T, m, m)))

            NK2 = NormalKernel(μ2, V2)
            DK2 = DiracKernel(μ2)

            # MISO
            μ3 = LinearMap(adjoint(randn(T, m)))
            V3 = one(real(T))

            NK3 = NormalKernel(μ3, V3)
            DK3 = DiracKernel(μ3)

            # Any
            IK = IdentityKernel()

            # marginalize
            for dist in (N, D), kernel in (NK2, DK2, IK)
                _test_pair_marginalize(dist, kernel)
            end

            # plus / minus
            v = randn(T, m)
            for dist in (N, D)
                @test mean(dist + v) == mean(v + dist) == mean(dist) + v
                @test mean(dist - v) == mean(dist) - v
                @test mean(v - dist) == v - mean(dist)
            end

            # multiplication
            C = randn(T, m, m)
            F = LinearMap(C)
            for dist in (N, D)
                @test C * dist == marginalize(dist, DiracKernel(F))
            end
        end

        @testset "marginalize | univariate" begin
            μ1 = randn(T)
            V1 = one(real(T))

            N = Normal(μ1, V1)
            D = Dirac(μ1)

            # SISO
            μ2 = LinearMap(randn(T))
            V2 = one(real(T))

            NK2 = NormalKernel(μ2, V2)
            DK2 = DiracKernel(μ2)

            # Any
            IK = IdentityKernel()

            # marginalize
            for dist in (N, D), kernel in (NK2, DK2, IK)
                _test_pair_marginalize(dist, kernel)
            end

            # plus / minus
            v = randn(T)
            for dist in (N, D)
                @test mean(dist + v) == mean(v + dist) == mean(dist) + v
                @test mean(dist - v) == mean(dist) - v
                @test mean(v - dist) == v - mean(dist)
            end

            # multiplication
            C = randn(T)
            F = LinearMap(C)
            for dist in (N, D)
                @test C * dist == marginalize(dist, DiracKernel(F))
            end
        end
    end
end
