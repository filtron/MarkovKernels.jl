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
        include("distributions/categorical_test.jl")
        include("distributions/dirac_test.jl")
        include("distributions/normal_test.jl")
        include("distributions/laplace_test.jl")
        #normal_plotting_test()
    end

    @testset "Kernels" begin
        include("kernels/normalkernel_test.jl")
        include("kernels/dirackernel_test.jl")
        include("kernels/stochasticmatrix_test.jl")
    end

    @testset "Likelihoods" begin
        include("likelihoods/categoricallikelihood_test.jl")
        include("likelihoods/flatlikelihood_test.jl")
        include("likelihoods/likelihood_test.jl")
        include("likelihoods/logquadratic_test.jl")
    end

    @testset "Binary Operations" begin
        include("binary_operations/compose_test.jl")
        include("binary_operations/marginalize_test.jl")
        include("binary_operations/invert_test.jl")
        include("binary_operations/posterior_test.jl")
        include("binary_operations/htransform_test.jl")
    end

    @testset "PSDMatrices" begin
        include("psdmatrices_test.jl")
    end

    @testset "Code quality (Aqua.jl)" begin
        #=
        There's a bunch of ambiguities with Random.rand for Markovkernels
        1) remove rand(::AbstractRNG, ::AbstractMarkovKernel, ::Any)
        2) define sample / drawfrom instead?
        =#
        Aqua.test_all(MarkovKernels; ambiguities = false)
    end

    if !occursin("DEV", string(VERSION))
        @testset "Code linting (JET.jl)" begin
            JET.test_package(MarkovKernels; target_defined_modules = true)
        end
    end

    @testset "Formatting (JuliaFormatter.jl)" begin
        @test JuliaFormatter.format(MarkovKernels; verbose = false, overwrite = false)
    end
end
