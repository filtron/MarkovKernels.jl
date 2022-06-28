# types for representing affine conditional means
abstract type AbstractAffineMap{T<:Number} <: AbstractConditionalMean{T}  end

#  T <: AbstractAffineMap implements slope and intercept
(M::AbstractAffineMap)(x) = slope(M)*x + intercept(M)

# AffineMap is default?
compose(M2::AbstractAffineMap,M1::AbstractAffineMap) = AffineMap(slope(M2)*slope(M1), slope(M2)*intercept(M1) + intercept(M2))
*(M2::AbstractAffineMap,M1::AbstractAffineMap) = compose(M2,M1)

# type for representing affine maps x ↦ prior + Φ*(x-pred)
struct AffineMap{T,U,V,S} <: AbstractConditionalMean{T}
    Φ::U
    prior::V
    pred::S
    AffineMap{T}(Φ,prior,pred) where T <: Number = new{T,typeof(Φ),typeof(prior),typeof(pred)}(Φ,prior,pred)
end

# constructors
AffineMap(Φ::AbstractMatrix{T}) where T <: Number = AffineMap{T}(Φ,nothing,nothing)

function AffineMap(Φ::AbstractMatrix,prior::AbstractVector)
    T = promote_type(eltype(Φ),eltype(prior))
    return AffineMap{T}( convert(AbstractMatrix{T},Φ), convert(AbstractVector{T},prior), nothing )
end

function AffineMap(Φ::AbstractMatrix,prior::AbstractVector,pred::AbstractVector)
    T = promote_type(eltype(Φ),eltype(prior),eltype(pred))
    return AffineMap{T}( convert(AbstractMatrix{T},Φ), convert(AbstractVector{T},prior), convert(AbstractVector{T},pred) )
end

# Seems convenient with constructors for nothings, there might be  a cleverer way...
AffineMap(Φ::AbstractMatrix,prior::Nothing,pred::Nothing) = AffineMap(Φ)
AffineMap(Φ::AbstractMatrix,prior::AbstractVector,pred::Nothing) = AffineMap(Φ,prior)

# functions
nout(M::AffineMap) = size(M.Φ,1)
nin(M::AffineMap)  = size(M.Φ,2)

has_pred(M::AffineMap) = !isnothing(M.pred)
has_prior(M::AffineMap) = !isnothing(M.prior)
islinear(M::AffineMap) = has_pred(M) == false && has_prior(M) == false ? true : false

slope(M::AffineMap) = M.Φ

function intercept(M::AffineMap)

    if islinear(M)
        return zeros(nout(M))
    end

    if has_prior(M)
        out = M.prior
    end

    if has_pred(M)
        out = out - M.Φ*M.pred
    end

    return out

end

function (M::AffineMap)(x)

    if islinear(M)
        return slope(M)*x
    end

    if has_pred(M)
        out = slope(M)*(x - M.pred)
    end

    if has_prior(M)
        out = out + M.prior
    end

    return out
end

function compose(M2::AffineMap,M1::AffineMap)

    Φ = slope(M2)*slope(M1)
    prior = has_prior(M1) ? M2(M1.prior) : nothing
    pred = M1.pred

    return AffineMap(Φ,prior,pred)

end

stein(Σ::Hermitian,M::AbstractAffineMap,Q::Hermitian) = Hermitian(slope(M)*Σ*slope(M)' + Q)


