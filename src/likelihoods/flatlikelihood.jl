"""
    FlatLikelihood

Type for representing flat likelihoods.
"""
struct FlatLikelihood <: AbstractLikelihood end

log(::FlatLikelihood, x) = zero(real(float(eltype(x))))
