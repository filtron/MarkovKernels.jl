using MarkovKernels
using Test, Aqua, JET
using LinearAlgebra
#using StaticArrays
#using Plots
import RecursiveArrayTools: recursivecopy, recursivecopy!

import LinearAlgebra: HermOrSym

include("matrix_test_utils.jl")

include("covariance_parameter_test.jl")

include("distributions/normal_test.jl")
include("distributions/dirac_test.jl")
#include("distributions/normal_plotting_test.jl")
include("distributions/particle_system_test.jl")

include("affinemap_test.jl")
include("kernels/normalkernel_test.jl")
include("kernels/dirackernel_test.jl")

include("likelihood_test.jl")
include("binary_operations/compose_test.jl")
include("binary_operations/marginalize_test.jl")
include("binary_operations/invert_test.jl")
include("binary_operations/bayes_rule_test.jl")

n = 1
m = 2

etypes = (Float64, Complex{Float64})

matrix_types = (Matrix,)
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
            particle_system_test()
        end
        #normal_plotting_test()
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

    @testset "Likelihood" begin
        for T in etypes
            likelihood_test(T, n, m, cov_types, matrix_types)
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

    @testset "marginalize" begin
        for T in etypes
            marginalize_test(T, n, m, cov_types, matrix_types)
            _test_marginalze_particle_system(T, n, m)
        end
    end

    @testset "invert" begin
        for T in etypes
            invert_test(T, n, m, cov_types, matrix_types)
        end
    end

    @testset "bayes_rule" begin
        for T in etypes
            bayes_rule_test(T, n, m, cov_types, matrix_types)
        end
    end

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MarkovKernels, piracies = false)
    end

    @testset "Code linting (JET.jl)" begin
        JET.test_package(MarkovKernels; target_defined_modules = true)
    end
end
