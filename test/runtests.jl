using MarkovKernels
using Test
using LinearAlgebra, StaticArrays
using Plots

import LinearAlgebra: HermOrSym

include("normal_test_utilities.jl")

include("covariance_parameter_test.jl")

include("distributions/normal_test.jl")
include("distributions/dirac_test.jl")
include("distributions/normal_plotting_test.jl")

include("affinemap_test.jl")
include("kernels/normalkernel_test.jl")
include("kernels/dirackernel_test.jl")
include("kernels/compose_test.jl")

include("likelihood_test.jl")

include("marginalise_test.jl")
include("invert_test.jl")

n = 1
m = 2

etypes = (Float64, Complex{Float64})
affine_types = (:LinearMap, :AffineMap, :AffineCorrector)

# in preparation for StaticArrays
cov_container = (:Matrix, :Diagonal)

cov_types = (:Matrix, :Diagonal, :Cholesky)

@testset "MarkovKernels.jl" begin
    @testset "CovarianceParameter" begin
        for T in etypes
            covariance_parameter_test(T)
        end
    end

    @testset "Distributions" begin
        for T in etypes
            normal_test(T, n, cov_types)
            dirac_test(T, n)
        end
        normal_plotting_test()
    end

    @testset "AffineMaps" begin
        for T in etypes
            affinemap_test(T, affine_types, n)
        end
    end

    @testset "Kernels" begin
        for T in etypes
            normalkernel_test(T, affine_types)
            affine_normalkernel_test(T, n, affine_types, cov_types)
            dirackernel_test(T, n, affine_types, cov_types)
        end
    end

    @testset "Likelihoods" begin
        for T in etypes, CT in cov_types
            likelihood_test(T, n, m, affine_types, CT)
        end
    end

    @testset "compose" begin
        for T in etypes, CT in cov_types
            compose_test(T, n, affine_types, CT)
        end
    end

    @testset "marginalise" begin
        for T in etypes, CT in cov_types
            marginalise_test(T, n, m, affine_types, CT)
        end
    end

    @testset "invert" begin
        for T in etypes, CT in cov_types
            invert_test(T, n, m, affine_types, CT)
        end
    end
end
