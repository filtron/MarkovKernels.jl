using MarkovKernels
using Test
using LinearAlgebra

include("normal_test.jl")
include("dirac_test.jl")

include("affinemap_test.jl")
include("normalkernel_test.jl")
include("dirackernel_test.jl")

n = 2

etypes = (Float64, Complex{Float64})

#amtypes = (:Linear,:Affine,:Corrector)
amtypes = (:Linear, :Affine)

@testset "MarkovKernels.jl" begin
    for T in etypes
        normal_test(T, n)
        dirac_test(T, n)
    end

    for T in etypes, MT in amtypes
        affinemap_test(T, MT, n)
    end

    for T in etypes
        normalkernel_test(T, n)
        dirackernel_test(T,n)
    end
end
