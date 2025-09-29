@safetestset "invert" begin
    using MarkovKernels, LinearAlgebra
    include("invert_test_utils.jl")

    m, n = 2, 3
    etys = (Float64, ComplexF64)

    for T in etys
        @testset "invert | Multivariate Dirac/Normal" begin
            μ = randn(T, n)
            V = Cholesky(UpperTriangular(ones(T, n, n)))

            N = Normal(μ, V)

            C1 = LinearMap(randn(T, m, n))
            R1 = Cholesky(UpperTriangular(ones(T, m, m)))

            NK1 = NormalKernel(C1, R1)
            DK1 = DiracKernel(C1)

            C2 = LinearMap(adjoint(randn(T, n)))
            R2 = one(real(T))

            NK2 = NormalKernel(C2, R2)
            DK2 = DiracKernel(C2)

            for K in (NK1, NK2, DK1, DK2)
                S, G, Π = _schur(_mat_or_num(V), slope(mean(K)), _cov(K))
                NC, KC = invert(N, K)

                @test cov(NC) ≈ S
                @test mean(NC) ≈ mean(K)(mean(N))
                @test _cov(KC) ≈ Π
                @test slope(mean(KC)) ≈ G
                @test intercept(mean(KC)) ≈ mean(N) - G * mean(K)(mean(N))
            end

            IK = IdentityKernel()
            NC, KC = invert(N, IK)

            @test N == NC
            @test IK == KC
        end

        @testset "invert | Univariate Dirac/Normal" begin
            μ = randn(T)
            V = one(real(T))

            N = Normal(μ, V)

            C1 = LinearMap(randn(T))
            R1 = one(real(T))

            NK1 = NormalKernel(C1, R1)
            DK1 = DiracKernel(C1)

            for K in (NK1, DK1)
                S, G, Π = _schur(_mat_or_num(V), slope(mean(K)), _cov(K))
                NC, KC = invert(N, K)

                @test cov(NC) ≈ S
                @test mean(NC) ≈ mean(K)(mean(N))
                @test abs(_cov(KC) - Π) ≤ 10 * eps(real(T))
                @test abs(slope(mean(KC)) - G) ≤ 10 * eps(real(T))
                @test abs(intercept(mean(KC)) - mean(N) + G * mean(K)(mean(N))) ≤
                      10 * eps(real(T))
            end

            IK = IdentityKernel()
            NC, KC = invert(N, IK)

            @test N == NC
            @test IK == KC
        end

        @testset "invert | StochasticMatrix" begin
            m, n = 2, 3
            etys = (Float64,)
            for T in etys
                π2 = exp.(randn(T, m))
                π2 = π2 / sum(π2)
                C2 = ProbabilityVector(π2)

                P2 = exp.(randn(T, m, m))
                P2 = P2 * Diagonal(1 ./ [sum(p) for p in eachcol(P2)])
                K2 = StochasticMatrix(P2)

                P3 = exp.(randn(T, n, m))
                P3 = P3 * Diagonal(1 ./ [sum(p) for p in eachcol(P3)])
                K3 = StochasticMatrix(P3)

                π4 = exp.(randn(T, n))
                π4 = π4 / sum(π4)
                C4 = ProbabilityVector(π4)

                P4 = exp.(randn(T, m, n))
                P4 = P4 * Diagonal(1 ./ [sum(p) for p in eachcol(P4)])
                K4 = StochasticMatrix(P4)

                C2C, K2C = invert(C2, K2)
                @test probability_vector(C2C) ≈ probability_vector(forward_operator(K2, C2))
                @test probability_matrix(K2C) ≈
                      Diagonal(π2) * adjoint(P2) * Diagonal(1 ./ probability_vector(C2C))

                C2C2, K3C = invert(C2, K3)
                @test probability_vector(C2C2) ≈
                      probability_vector(forward_operator(K3, C2))
                @test probability_matrix(K3C) ≈
                      Diagonal(π2) * adjoint(P3) * Diagonal(1 ./ probability_vector(C2C2))

                C4C, K4C = invert(C4, K4)
                @test probability_vector(C4C) ≈ probability_vector(forward_operator(K4, C4))
                @test probability_matrix(K4C) ≈
                      Diagonal(π4) * adjoint(P4) * Diagonal(1 ./ probability_vector(C4C))
            end
        end
    end
end
