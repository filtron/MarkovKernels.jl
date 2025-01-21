@safetestset "htransform" begin
    using MarkovKernels, LinearAlgebra

    n, m = 2, 3
    etys = (Float64,)

    @testset "htransform | StochasticMatrix / CategoricalLikelihood" begin
        for T in etys
            π = exp.(randn(T, m))
            π = π / sum(π)

            C0 = Categorical(π)

            Pxx = exp.(randn(T, m, m))
            Pxx = Pxx * Diagonal(1 ./ [sum(p) for p in eachcol(Pxx)])
            Kxx = StochasticMatrix(Pxx)

            Pyx = exp.(randn(T, n, m))
            Pyx = Pyx * Diagonal(1 ./ [sum(p) for p in eachcol(Pyx)])
            Kyx = StochasticMatrix(Pyx)
            y = rand(1:n)

            L = Likelihood(Kyx, y)

            # forward / backward
            C1, Bxx = invert(C0, Kxx)
            C1, ll_fb = posterior_and_loglike(C1, L)
            C0_fb, Fxx_fb = invert(C1, Bxx)

            # backward / forward
            Fxx_bf, L0 = htransform(Kxx, L)
            C0_bf, ll_bf = posterior_and_loglike(C0, L0)

            @test ll_fb ≈ ll_bf
            @test C0_fb ≈ C0_bf
            @test Fxx_fb ≈ Fxx_bf

            L2 = FlatLikelihood()
            @test all(splat(isequal), zip(htransform(Kxx, L2), (Kxx, L2)))
        end
    end

    @testset "htransform | Multivariate NormalKernel" begin
        n, m = 2, 3
        etys = (Float64, ComplexF64)
        for T in etys
            Φ = LinearMap(randn(T, m, m))
            Σ = Cholesky(UpperTriangular(ones(T, m, m)))
            K = NormalKernel(Φ, Σ)
            x0 = randn(T, m)
            x = rand(condition(K, x0))

            C1 = LinearMap(randn(n, m))
            R1 = Cholesky(UpperTriangular(ones(n, n)))
            K1 = NormalKernel(C1, R1)
            y1 = rand(condition(K1, x))
            L1 = Likelihood(K1, y1)

            C2 = LinearMap(adjoint(randn(m)))
            R2 = exp(randn(real(T)))
            K2 = NormalKernel(C2, R2)
            y2 = rand(condition(K2, x))
            L2 = Likelihood(K2, y2)

            R3 = exp(randn(real(T))) * I
            K3 = NormalKernel(C1, I)
            y3 = rand(condition(K3, x))
            L3 = Likelihood(K3, y3)

            K4 = DiracKernel(C1)
            y4 = rand(condition(K4, x))
            L4 = Likelihood(K4, y4)

            K5 = DiracKernel(C2)
            y5 = rand(condition(K5, x))
            L5 = Likelihood(K5, y5)

            Ls = (L1, L2, L3, L4, L5)
            for L in Ls
                N = condition(K, x0)
                NCgt, llgt = posterior_and_loglike(N, L)

                Knew, Lnew = htransform(K, L)
                NC = condition(Knew, x0)
                ll = log(Lnew, x0)

                @test mean(NC) ≈ mean(NCgt)
                @test cov(NC) ≈ cov(NCgt)
                @test ll ≈ llgt
            end

            L = FlatLikelihood()
            @test all(splat(isequal), zip(htransform(K, L), (K, L)))
        end
    end

    @testset "htransform | Univariate NormalKernel" begin
        etys = (Float64, ComplexF64)
        for T in etys
            Φ = LinearMap(randn(T))
            Σ = exp(randn(real(T)))
            K = NormalKernel(Φ, Σ)
            x0 = randn(T)
            x = rand(condition(K, x0))

            C1 = LinearMap(randn(T))
            R1 = exp(randn(real(T)))
            K1 = NormalKernel(C1, R1)
            y1 = rand(condition(K1, x))
            L1 = Likelihood(K1, y1)

            R2 = exp(randn(real(T))) * I
            K2 = NormalKernel(C1, R2)
            y2 = rand(condition(K2, x))
            L2 = Likelihood(K2, y2)

            # exclude L2 for now since \(::UniformScaling, ::Number) is not implemented?
            Ls = (L1, L2)
            for L in Ls
                N = condition(K, x0)
                NCgt, llgt = posterior_and_loglike(N, L)

                Knew, Lnew = htransform(K, L)
                NC = condition(Knew, x0)
                ll = log(Lnew, x0)

                @test mean(NC) ≈ mean(NCgt)
                @test cov(NC) ≈ cov(NCgt)
                @test ll ≈ llgt
            end

            L = FlatLikelihood()
            @test all(splat(isequal), zip(htransform(K, L), (K, L)))
        end
    end
end
