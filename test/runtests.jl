using MarkovKernels
using Test, Aqua, JET, JuliaFormatter
using SafeTestsets

using LinearAlgebra
#using Plots
import RecursiveArrayTools: recursivecopy, recursivecopy!

#include("distributions/normal_plotting_test.jl")

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

    @testset "Binary Operations" begin
        include("binary_operations/compose_test.jl")
        include("binary_operations/marginalize_test.jl")
        include("binary_operations/invert_test.jl")
        include("binary_operations/posterior_test.jl")
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
