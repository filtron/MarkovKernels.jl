using MarkovKernels
using Test
using LinearAlgebra


include("normal_test.jl")

n = 2

etypes = (Float64,Complex{Float64})

@testset "MarkovKernels.jl" begin

    for T in etypes
    normal_test(T,n)
    end

end
