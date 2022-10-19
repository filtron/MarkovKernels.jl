using MarkovKernels
using Test
using LinearAlgebra
using Plots

include("normal_test_utilities.jl")

include("normal_test.jl")
include("dirac_test.jl")

include("affinemap_test.jl")
include("normalkernel_test.jl")
include("dirackernel_test.jl")
include("likelihood_test.jl")

include("normal_plotting_test.jl")

n = 2
m = 3

etypes = (Float64, Complex{Float64})

amtypes = (:Linear, :Affine)

affine_types = (:LinearMap, :AffineMap, :AffineCorrector)
cov_types = (:Matrix, :Diagonal, :UniformScaling, :Cholesky)

@testset "MarkovKernels.jl" begin
    for T in etypes
        normal_test(T, n, cov_types)
        dirac_test(T, n)
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

    normal_plotting_test()
end
