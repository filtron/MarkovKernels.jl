function affinemap_test(T,MT,n)


Φ1 = randn(T,n,n)
Φ2 = randn(T,n,n)

RV = randn(T,n,n)
V = Hermitian(RV'*RV)

RQ = randn(T,n,n)
Q = Hermitian(RQ'*RQ)

x = randn(T,n)

# composition
Φ3 = Φ2*Φ1

if MT == LinearMap
    M1 = LinearMap(Φ1)
    M2 = LinearMap(Φ2)
    b1 = zeros(T,n)
    b2 = zeros(T,n)
    b3 = zeros(T,n)
elseif MT == AffineMap
    b1 = randn(T,n)
    b2 = randn(T,n)
    M1 = AffineMap(Φ1,b1)
    M2 = AffineMap(Φ2,b2)
    b3 = Φ2*b1 + b2
end

stein1 = Hermitian(Φ1*V*Φ1' + Q)


M3 = compose(M2,M1)

@testset "AffineMap | $(T) | $(MT)" begin

    @test eltype(M1) == T

    @test slope(M1) == Φ1
    @test intercept(M1) == b1
    @test stein(V,M1,Q) ≈ stein1
    @test M1(x) ≈ slope(M1)*x + intercept(M1)

    @test slope(M3) ≈ Φ3
    @test intercept(M3) ≈ b3

end





end