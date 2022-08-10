function affinemap_test(T, MT, n)
    Φ1 = randn(T, n, n)
    Φ2 = randn(T, n, n)

    x = randn(T, n)

    # composition
    Φ3 = Φ2 * Φ1

    if MT === :Linear
        M1 = AffineMap(Φ1)
        M2 = AffineMap(Φ2)
        prior1 = zeros(T, n)
        prior2 = zeros(T, n)
        prior3 = zeros(T, n)
    elseif MT == :Affine
        prior1 = randn(T, n)
        prior2 = randn(T, n)
        M1 = AffineMap(Φ1, prior1)
        M2 = AffineMap(Φ2, prior2)
        prior3 = Φ2 * prior1 + prior2
    end

    M3 = compose(M2, M1)

    @testset "AffineMap | $(T) | $(MT)" begin
        @test eltype(M1) == T

        @test slope(M1) == Φ1
        @test intercept(M1) == prior1
        @test M1(x) ≈ slope(M1) * x + intercept(M1)

        @test slope(M3) ≈ Φ3
        @test intercept(M3) ≈ prior3
    end
end
