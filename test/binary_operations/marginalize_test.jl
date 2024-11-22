@safetestset "marginalize" begin
    using MarkovKernels, LinearAlgebra
    include("marginalize_test_utils.jl")

    m = 2
    etys = (Float64, ComplexF64)

    for T in etys
        @testset "marginalize | Multivariate Dirac/Normal" begin
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

        @testset "marginalize | Univariate Dirac/Normal" begin
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

        @testset "marginalize | Stochastic Matrix" begin
            etys = (Float64,)
            m, n = 2, 3
            for T in etys
                π2 = exp.(randn(T, m))
                π2 = π2 / sum(π2)
                C2 = Categorical(π2)

                P2 = exp.(randn(T, m, m))
                P2 = P2 * Diagonal(1 ./ [sum(p) for p in eachcol(P2)])
                K2 = StochasticMatrix(P2)

                P3 = exp.(randn(T, n, m))
                P3 = P3 * Diagonal(1 ./ [sum(p) for p in eachcol(P3)])
                K3 = StochasticMatrix(P3)

                π4 = exp.(randn(T, n))
                π4 = π4 / sum(π4)
                C4 = Categorical(π4)

                P4 = exp.(randn(T, m, n))
                P4 = P4 * Diagonal(1 ./ [sum(p) for p in eachcol(P4)])
                K4 = StochasticMatrix(P4)

                @test probability_vector(marginalize(C2, K2)) ≈ P2 * π2
                @test probability_vector(marginalize(C2, K3)) ≈ P3 * π2
                @test probability_vector(marginalize(C4, K4)) ≈ P4 * π4
            end
        end
    end
end
