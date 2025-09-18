function ldiv!(d, a::LinearMap)
    ldiv!(d, slope(a))
    return a
end

function ldiv!(d, a::AffineMap)
    ldiv!(d, slope(a))
    ldiv!(d, intercept(a))
    return a
end

function ldiv!(d, a::AffineCorrector)
    ldiv!(d, a.A)
    ldiv!(d, a.b)
    return a
end

ldiv!(aout::AbstractAffineMap, d, a::AbstractAffineMap) = ldiv!(d, copy!(aout, a))

function mul!(c::LinearMap, A, b::LinearMap, α, β)
    Sc, Sb = slope(c), slope(b)
    mul!(Sc, A, Sb, α, β)
    return c
end

function mul!(c::AffineMap, A, b::AffineMap, α, β)
    Sc, Sb = slope(c), slope(b)
    Ic, Ib = intercept(c), intercept(b)
    mul!(Sc, A, Sb, α, β)
    mul!(Ic, A, Ib, α, β)
    return c
end

function mul!(c::AffineMap, A, b::LinearMap, α, β)
    Sc, Sb = slope(c), slope(b)
    Ic = intercept(c)
    mul!(Sc, A, Sb, α, β)
    lmul!(β, Ic)
    return c
end
