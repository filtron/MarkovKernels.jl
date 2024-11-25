@safetestset "FlatLikelihood" begin
    using MarkovKernels, LinearAlgebra
    etys = (Float64, ComplexF64)
    n = 2

    L1 = FlatLikelihood()
    L2 = FlatLikelihood()
    for T in etys
        x = randn(T, 1)

        @test_nowarn FlatLikelihood()
        @test L1 === L2
        @test log(L1, x) == zero(real(eltype(x)))
        @test typeof(log(L1, x)) == typeof(zero(real(eltype(x))))
    end
end
