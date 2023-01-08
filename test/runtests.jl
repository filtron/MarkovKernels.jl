using MarkovKernels
using Test
using LinearAlgebra, StaticArrays
using Plots

import LinearAlgebra: HermOrSym

include("matrix_test_utils.jl")

include("covariance_parameter_test.jl")

include("distributions/normal_test.jl")
include("distributions/dirac_test.jl")
include("distributions/normal_plotting_test.jl")
include("distributions/mixture_test.jl")
include("distributions/particle_system_test.jl")

include("affinemap_test.jl")
include("kernels/normalkernel_test.jl")
include("kernels/dirackernel_test.jl")

include("loglike_test.jl")
include("binary_operations/compose_test.jl")
include("binary_operations/marginalise_test.jl")
include("binary_operations/invert_test.jl")

n = 1
m = 2

etypes = (Float64, Complex{Float64})

matrix_types = (Matrix, SMatrix)
affine_types = (LinearMap, AffineMap, AffineCorrector)
cov_types = (HermOrSym, Cholesky)

@testset "MarkovKernels.jl" begin
    @testset "CovarianceParameter" begin
        for T in etypes
            covariance_parameter_test(T, cov_types, matrix_types)
        end
    end

    @testset "Distributions" begin
        for T in etypes
            normal_test(T, n, cov_types, matrix_types)
            dirac_test(T, n)
            mixture_test()
            particle_system_test()
        end
        normal_plotting_test()
    end

    @testset "AffineMaps" begin
        for T in etypes
            affinemap_test(T, n, affine_types, matrix_types)
        end
    end

    @testset "Kernels" begin
        for T in etypes
            normalkernel_test(T)
            affine_normalkernel_test(T, n, cov_types, matrix_types)
            dirackernel_test(T, n, matrix_types)
        end
    end

    @testset "LogLike" begin
        for T in etypes
            loglike_test(T, n, m, cov_types, matrix_types)
        end
    end

    @testset "compose" begin
        C = [1.0 -1.0]
        K1 = DiracKernel(C)
        variance(x) = fill(exp.(x)[1], 1, 1)
        K2 = NormalKernel(zeros(1, 1), variance)

        @testset "compose | $(nameof(typeof(K2))) | $(nameof(typeof(K1)))" begin
            m_kernel = compose(K2, K1)
            x = randn(2)

            @test mean(m_kernel)(x) ≈ zeros(1)
            @test cov(m_kernel)(x) ≈ variance(C * x)
        end

        for T in etypes
            compose_test(T, n, cov_types, matrix_types)
        end
    end

    @testset "marginalise" begin
        for T in etypes
            marginalise_test(T, n, m, cov_types, matrix_types)
            _test_marginalse_particle_system(T, n, m)
        end
    end

    @testset "invert" begin
        for T in etypes
            invert_test(T, n, m, cov_types, matrix_types)
        end
    end
end
