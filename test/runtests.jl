using MarkovKernels
using Test
using LinearAlgebra


include("normal_test.jl")

n = 2

etypes = (Float64,Complex{Float64})

parametrisations = (Usual,)

@testset "MarkovKernels.jl" begin

    for T in etypes, P in parametrisations
    normal_test(T,P,n)
    end

end
