@safetestset "compose" begin
    using MarkovKernels, LinearAlgebra
    include("compose_test_utils.jl")

    m = 2
    etys = (Float64, ComplexF64)

    @testset "compose | AffineHeteroskedasticNormalKernel | DiracKernel" begin
        for T in etys

            # MISO
            F1 = LinearMap(adjoint([one(T), -one(T)]))
            K1 = DiracKernel(F1)

            # SISO
            F2 = LinearMap(zero(T))
            v2(x) = exp(real(x))
            K2 = NormalKernel(F2, v2)
            K21 = compose(K2, K1)

            x = randn(T, 2)

            @test mean(K21)(x) ≈ zero(T)
            @test cov(K21)(x) ≈ v2(F1(x))
        end
    end

    @testset "compose | AffineHomoskedasticNormalKernel | DiracKernel" begin
        for T in etys
            F1 = LinearMap(randn(T, m, m))
            V1 = Cholesky(UpperTriangular(ones(T, m, m)))

            F2 = LinearMap(randn(T, m, m))
            V2 = Cholesky(UpperTriangular(ones(T, m, m)))

            x = randn(T, m)

            NK1 = NormalKernel(F1, V1)
            NK2 = NormalKernel(F2, V2)

            DK1 = DiracKernel(F1)
            DK2 = DiracKernel(F2)

            IK = IdentityKernel()

            kernels = (NK1, NK2, DK1, DK2, IK)

            for K1 in kernels, K2 in kernels
                _test_pair_compose(K1, K2)
            end
        end
    end
end
