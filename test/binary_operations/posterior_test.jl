@safetestset "posterior" begin
    using MarkovKernels, LinearAlgebra
    include("invert_test_utils.jl")

    m, n = 2, 3
    etys = (Float64, ComplexF64)

    for T in etys
        @testset "posterior | Multivariate Dirac/Normal" begin
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

            for L in (L1, L3)
                Lq = LogQuadraticLikelihood(L)

                @test all(
                    splat(isapprox),
                    zip(posterior_and_loglike(N, Lq), posterior_and_loglike(N, L)),
                )
                @test isapprox(posterior(N, Lq), posterior(N, L))
            end

            L = FlatLikelihood()
            P1, ll = posterior_and_loglike(N, L)
            P2 = posterior(N, L)

            @test P1 == P2 == N
            @test iszero(ll)
        end

        @testset "posterior | Univariate Dirac/Normal" begin
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

            Lq = LogQuadraticLikelihood(L1)
            @test all(
                splat(isapprox),
                zip(posterior_and_loglike(N, Lq), posterior_and_loglike(N, L1)),
            )
            @test isapprox(posterior(N, Lq), posterior(N, L1))

            L = FlatLikelihood()
            P1, ll = posterior_and_loglike(N, L)
            P2 = posterior(N, L)

            @test P1 == P2 == N
            @test iszero(ll)
        end
    end

    m, n = 2, 3
    etys = (Float64,)

    for T in etys
        @testset "StochasticMatrix / CategoricalLikelihood" begin
            π1 = exp.(randn(T, m))
            π1 = π1 / sum(π1)
            C1 = Categorical(π1)

            P1 = exp.(randn(T, m, m))
            P1 = P1 * Diagonal(1 ./ [sum(p) for p in eachcol(P1)])
            K1 = StochasticMatrix(P1)

            π2 = exp.(randn(T, n))
            π2 = π2 / sum(π2)
            C2 = Categorical(π2)

            P2 = exp.(randn(T, m, n))
            P2 = P2 * Diagonal(1 ./ [sum(p) for p in eachcol(P2)])
            K2 = StochasticMatrix(P2)

            y = rand(1:m)
            L1 = Likelihood(K1, y)
            L2 = Likelihood(K2, y)

            for (C, L) in zip((C1, C2), (L1, L2))
                π = probability_vector(C)
                K, y = measurement_model(L), measurement(L)
                P = probability_matrix(K)

                ls = P[y, :]
                my = dot(ls, π)
                πgt = π .* ls / my
                llgt = log(my)

                CL = CategoricalLikelihood(L)

                CC1, ll1 = posterior_and_loglike(C, L)
                CC2 = posterior(C, L)
                CC3, ll2 = posterior_and_loglike(C, CL)
                CC4 = posterior(C, CL)

                @test ll1 ≈ ll2 ≈ llgt
                @test probability_vector(CC1) ≈
                      probability_vector(CC2) ≈
                      probability_vector(CC3) ≈
                      probability_vector(CC4) ≈
                      πgt
            end
        end
    end
end
