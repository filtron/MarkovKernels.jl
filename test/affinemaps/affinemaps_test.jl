@testset "AffineMaps" begin
    include("test_utils.jl")

    include("linearmap_test.jl")
    include("affinemap_test.jl")
    include("affinecorrector_test.jl")
    include("mixed_ops_test.jl")
end
