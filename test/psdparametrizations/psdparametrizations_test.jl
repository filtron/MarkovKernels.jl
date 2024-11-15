@testset "PSDParametrizations" begin
    include("test_utils.jl")
    include("scalar_test.jl")
    include("selfadjoint_test.jl")
    include("cholesky_test.jl")
end
