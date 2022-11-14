using MarkovKernels
using Test
using LinearAlgebra, StaticArrays
using Plots

import LinearAlgebra: HermOrSym

include("matrix_test_utils.jl")
#include("normal_test_utilities.jl")

include("covariance_parameter_test.jl")

include("distributions/normal_test.jl")
include("distributions/dirac_test.jl")
include("distributions/normal_plotting_test.jl")

include("affinemap_test.jl")
include("kernels/normalkernel_test.jl")
include("kernels/dirackernel_test.jl")
include("kernels/compose_test.jl")

include("loglike_test.jl")

include("marginalise_test.jl")
include("invert_test.jl")

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
        for T in etypes
            compose_test(T, n, cov_types, matrix_types)
        end
    end

    @testset "marginalise" begin
        for T in etypes
            marginalise_test(T, n, m, cov_types, matrix_types)
        end
    end

    @testset "invert" begin
        for T in etypes
            invert_test(T, n, m, cov_types, matrix_types)
        end
    end
end
