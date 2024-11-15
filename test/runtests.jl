using MarkovKernels
using Test, Aqua, JET, JuliaFormatter
using LinearAlgebra
#using Plots
import RecursiveArrayTools: recursivecopy, recursivecopy!

import LinearAlgebra: HermOrSym

include("matrix_test_utils.jl")

#include("distributions/normal_plotting_test.jl")

include("binary_operations/invert_test.jl")
include("binary_operations/bayes_rule_test.jl")

n = 1
m = 2

etypes = (Float64, Complex{Float64})

matrix_types = (Matrix,)
cov_types = (HermOrSym, Cholesky)

@testset "MarkovKernels.jl" begin
    include("psdparametrizations/psdparametrizations_test.jl")
    include("affinemaps/affinemaps_test.jl")

    @testset "Distributions" begin
        include("distributions/dirac_test.jl")
        include("distributions/normal_test.jl")
        include("distributions/particle_system_test.jl")
        #normal_plotting_test()
    end

    @testset "Kernels" begin
        include("kernels/normalkernel_test.jl")
        include("kernels/dirackernel_test.jl")
    end

    @testset "Likelihoods" begin
        include("likelihood_test.jl")
    end

    @testset "compose" begin
        include("binary_operations/compose_test.jl")
    end

    @testset "marginalize" begin
        include("binary_operations/marginalize_test.jl")
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

    @testset "PSDMatrices" begin
        include("psdmatrices_test.jl")
    end

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MarkovKernels; ambiguities = false)
    end

    @testset "Code linting (JET.jl)" begin
        JET.test_package(MarkovKernels; target_defined_modules = true)
    end

    @testset "Formatting (JuliaFormatter.jl)" begin
        @test JuliaFormatter.format(MarkovKernels; verbose = false, overwrite = false)
    end
end
