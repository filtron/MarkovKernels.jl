using MarkovKernels
using Test
using LinearAlgebra


include("normal_test.jl")
include("affinemap_test.jl")

n = 2

etypes = (Float64,Complex{Float64})

amtypes = (LinearMap,AffineMap,AffineCorrector)

@testset "MarkovKernels.jl" begin

    for T in etypes
        normal_test(T,n)
    end

    for T in etypes, MT in amtypes
        affinemap_test(T,MT,n)
    end

end
