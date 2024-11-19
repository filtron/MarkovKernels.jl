@safetestset "posterior" begin
    using MarkovKernels, LinearAlgebra
    include("invert_test_utils.jl")

    m, n = 2, 3
    etys = (Float64, ComplexF64)

    for T in etys
        @testset "posterior | multivariate" begin
            μ = randn(T, n)
            V = Cholesky(UpperTriangular(ones(T, n, n)))

            N = Normal(μ, V)

            C1 = LinearMap(randn(T, m, n))
            R1 = Cholesky(UpperTriangular(ones(T, m, m)))
            y1 = randn(T, m)

            NK1 = NormalKernel(C1, R1)
            DK1 = DiracKernel(C1)

            C2 = LinearMap(adjoint(randn(T, n)))
            R2 = one(real(T))
            y2 = randn(T)

            NK2 = NormalKernel(C2, R2)
            DK2 = DiracKernel(C2)

            L1 = Likelihood(NK1, y1)
            L2 = Likelihood(DK1, y1)
            L3 = Likelihood(NK2, y2)
            L4 = Likelihood(DK2, y2)
            Ls = (L1, L2, L3, L4)

            for L in Ls
                K, y = measurement_model(L), measurement(L)

                S, G, Π = _schur(_mat_or_num(V), slope(mean(K)), _cov(K))

                P1, ll = posterior_and_loglike(N, L)
                P2 = posterior(N, L)

                @test mean(P1) ≈ mean(P2) ≈ mean(N) + G * (y - mean(K)(mean(N)))
                @test cov(P1) ≈ cov(P2) ≈ Π
            end

            L = FlatLikelihood()
            P1, ll = posterior_and_loglike(N, L)
            P2 = posterior(N, L)

            @test P1 == P2 == N
            @test iszero(ll)
        end

        @testset "posterior | univariate" begin
            μ = randn(T)
            V = one(real(T))

            N = Normal(μ, V)

            C1 = LinearMap(randn(T))
            R1 = one(real(T))
            y1 = randn(T)

            NK1 = NormalKernel(C1, R1)
            DK1 = DiracKernel(C1)

            L1 = Likelihood(NK1, y1)
            L2 = Likelihood(DK1, y1)

            for L in (L1, L2)
                K, y = measurement_model(L), measurement(L)

                S, G, Π = _schur(_mat_or_num(V), slope(mean(K)), _cov(K))

                P1, ll = posterior_and_loglike(N, L)
                P2 = posterior(N, L)

                @test mean(P1) ≈ mean(P2) ≈ mean(N) + G * (y - mean(K)(mean(N)))
                @test isapprox(cov(P1), Π, atol = 10 * eps(real(T))) &&
                      isapprox(cov(P2), Π, atol = 10 * eps(real(T)))
            end

            L = FlatLikelihood()
            P1, ll = posterior_and_loglike(N, L)
            P2 = posterior(N, L)

            @test P1 == P2 == N
            @test iszero(ll)
        end
    end
end
