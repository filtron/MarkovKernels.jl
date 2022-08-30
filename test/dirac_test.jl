
function dirac_test(T, n)
    μ1 = randn(T, n)
    D1 = Dirac(μ1)
    D12 = Dirac(μ1)

    @testset "Dirac | $(T) " begin

        @test D1 == D12
        @test mean(D1) == μ1
        @test cov(D1) == zeros(T,n,n)
        @test var(D1) == zeros(T,n)
        @test std(D1) == zeros(T,n)
        @test rand(D1) == mean(D1)

        @test eltype(var(D1)) <: Real
        @test eltype(std(D1)) <: Real
    end
end
