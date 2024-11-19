@safetestset "invert" begin
    using MarkovKernels, LinearAlgebra
    include("invert_test_utils.jl")

    m, n = 2, 3
    etys = (Float64, ComplexF64)

    for T in etys
        @testset "invert | multivariate" begin
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

        @testset "invert | univariate" begin
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
    end
end
