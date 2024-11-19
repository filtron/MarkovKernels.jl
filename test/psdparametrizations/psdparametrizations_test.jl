@safetestset "PSDParametrizations" begin
    using MarkovKernels
    using LinearAlgebra

    include("test_utils.jl")
    include("scalar_test.jl")
    include("selfadjoint_test.jl")
    include("cholesky_test.jl")
end
