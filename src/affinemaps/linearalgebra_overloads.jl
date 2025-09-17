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


#=
mul!
=#