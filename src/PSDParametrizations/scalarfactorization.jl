struct ScalarFactorization{T<:Real,A}
    σ::A
end

ScalarFactorization(σ::Real) = ScalarFactorization{typeof(σ),typeof(σ)}(σ)
ScalarFactorization(σ::Base.RefValue{T}) where {T<:Real} =
    ScalarFactorization{T,typeof(σ)}(σ)

Base.eltype(::ScalarFactorization{T}) where {T} = T

psdcheck(::ScalarFactorization) = IsPSD()
rsqrt(Σ::ScalarFactorization) = Σ.σ[]

convert_psd_eltype(::Type{T}, Σ::ScalarFactorization) where {T} =
    ScalarFactorization(convert(real(T), rsqrt(Σ)))

stein(Σ::ScalarFactorization, Φ::Number) = ScalarFactorization(abs(adjoint(Φ) * rsqrt(Σ)))

function stein(Σ::ScalarFactorization, Φ::Number, Q::ScalarFactorization)
    σ1 = rsqrt(Q)
    σ2 = adjoint(Φ) * rsqrt(Σ)
    _, r = givens(σ1, σ2, 1, 2)
    Π = ScalarFactorization(abs(r))
    return Π
end

function schur_reduce(Π::ScalarFactorization, C::Number)
    # this probably breaks if iszero(C) returns true
    S = ScalarFactorization(abs(C * lsqrt(Π)))
    K = adjoint(C) / abs2(C)
    Σ = ScalarFactorization(zero(lsqrt(Π)))
    return S, K, Σ
end

function schur_reduce(Π::ScalarFactorization, C::Number, R::ScalarFactorization)
    # this probably breaks if iszero(C) && iszero(R) returns true

    G, r = givens(rsqrt(R), rsqrt(Π) * adjoint(C), 1, 2)
    c, s = G.c, G.s

    # pre_array = [rsqrt(R) 0; rsqrt(Π) * adjoint(C) rsqrt(Π)]
    # Qadj = [c s; -conj(s) c]
    # istriu(Qadj * pre_array) = true, i.e. qr factorization
    S = ScalarFactorization(abs(r))
    Σ = ScalarFactorization(abs(c * rsqrt(Π)))
    K = conj(s * rsqrt(Π)) / lsqrt(S)
    return S, K, Σ
end

function Base.show(io::IO, Σ::ScalarFactorization)
    println(io, summary(Σ))
    print(io, "√Σ = ")
    show(io, (Σ.σ))
end
