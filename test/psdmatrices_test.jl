@testset "PSDMatrices | constructor" begin 
    using PSDMatrices
    MarkovKernels.Normal(μ::AbstractVector{T}, Σ::PSDMatrix{T}) where {T} = Normal{T,typeof(μ),typeof(Σ)}(μ,Σ)

    μ = zeros(2)
    Σ = PSDMatrix(randn(2, 2))
    @test_nowarn Normal(μ, Σ)
end 