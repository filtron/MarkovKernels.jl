using MarkovKernels
using Test
using LinearAlgebra
using Plots

include("normal_test_utilities.jl")

include("normal_test.jl")
include("dirac_test.jl")
include("normal_plotting_test.jl")

include("affinemap_test.jl")

include("normalkernel_test.jl")
include("dirackernel_test.jl")
include("likelihood_test.jl")

include("compose_test.jl")
include("marginalise_test.jl")
include("invert_test.jl")

n = 1
m = 2

etypes = (Float64, Complex{Float64})
affine_types = (:LinearMap, :AffineMap, :AffineCorrector)
cov_types = (:Matrix, :Diagonal, :UniformScaling, :Cholesky)

@testset "MarkovKernels.jl" begin
    @testset "Distributions" begin
        for T in etypes
            normal_test(T, n, cov_types)
            dirac_test(T, n)
        end
        normal_plotting_test()
    end

    for T in etypes
        affinemap_test(T, affine_types, n)
    end

    for T in etypes
        normalkernel_test(T, affine_types)
        affine_normalkernel_test(T, n, affine_types, cov_types)
        dirackernel_test(T, n, affine_types, cov_types)
        likelihood_test(T, n, m)
    end

    @testset "compose" begin
        for T in etypes
            compose_test(T, n, affine_types, cov_types)
        end
    end

    @testset "marginalise" begin
        for T in etypes
            marginalise_test(T, n, m, affine_types, cov_types)
        end
    end

    @testset "invert" begin
        for T in etypes
            invert_test(T, n, m, affine_types, cov_types)
        end
    end
end
