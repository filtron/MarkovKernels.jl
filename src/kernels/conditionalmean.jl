
abstract type AbstractConditionalMean{T<:Number} end

eltype(M::AbstractConditionalMean{T}) where {T} = T

# list of things all ConditionalMeans must have?
nin(μ::AbstractConditionalMean) = nin(μ)
nout(μ::AbstractConditionalMean) = nout(μ)
(μ::AbstractConditionalMean)(x) = μ(x)

# need to annotate number of inputs/ outputs here I think...
struct ConditionalMean{T,U,V} <: AbstractConditionalMean{T}
    f::U
    jac::V
    ConditionalMean{T}(f, jac) where {T<:Number} = new{T,typeof(f),typeof(jac)}(f, jac)
end

ConditionalMean{T}(μ) where {T} = ConditionalMean{T}(μ, nothing)
ConditionalMean(μ) = ConditionalMean{eltype(μ)}(μ, nothing)
(M::ConditionalMean)(x) = μ(x)

# types for representing affine conditional means
abstract type AbstractAffineMap{T<:Number} <: AbstractConditionalMean{T} end

#  T <: AbstractAffineMap implements slope and intercept
(M::AbstractAffineMap)(x) = slope(M) * x + intercept(M)

# AffineMap is default?
compose(M2::AbstractAffineMap, M1::AbstractAffineMap) =
    AffineMap(slope(M2) * slope(M1), slope(M2) * intercept(M1) + intercept(M2))
*(M2::AbstractAffineMap, M1::AbstractAffineMap) = compose(M2, M1)

# type for representing affine maps x ↦ prior + Φ*(x-pred)
struct AffineMap{T,U,V,S} <: AbstractAffineMap{T}
    Φ::U
    prior::V
    pred::S
    AffineMap{T}(Φ, prior, pred) where {T<:Number} =
        new{T,typeof(Φ),typeof(prior),typeof(pred)}(Φ, prior, pred)
end

# constructors
AffineMap(Φ::AbstractMatrix{T}) where {T<:Number} = AffineMap{T}(Φ, nothing, nothing)

function AffineMap(Φ::AbstractMatrix, prior::AbstractVector)
    T = promote_type(eltype(Φ), eltype(prior))
    return AffineMap{T}(
        convert(AbstractMatrix{T}, Φ),
        convert(AbstractVector{T}, prior),
        nothing,
    )
end

function AffineMap(Φ::AbstractMatrix, prior::AbstractVector, pred::AbstractVector)
    T = promote_type(eltype(Φ), eltype(prior), eltype(pred))
    return AffineMap{T}(
        convert(AbstractMatrix{T}, Φ),
        convert(AbstractVector{T}, prior),
        convert(AbstractVector{T}, pred),
    )
end

==(M1::AffineMap, M2::AffineMap) =
    M1.Φ == M2.Φ && M1.prior == M2.prior && M1.pred == M2.pred
function similar(M::AffineMap)
    Φ = similar(M.Φ)
    prior = prior === nothing ? nothing : similar(prior)
    pred = pred === nothing ? nothing : similar(pred)

    return AffineMap(Φ, prior, pred)
end

# Seems convenient with constructors for nothings, there might be  a cleverer way...
AffineMap(Φ::AbstractMatrix, prior::Nothing, pred::Nothing) = AffineMap(Φ)
AffineMap(Φ::AbstractMatrix, prior::AbstractVector, pred::Nothing) = AffineMap(Φ, prior)

# functions
nout(M::AffineMap) = size(M.Φ, 1)
nin(M::AffineMap) = size(M.Φ, 2)

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
        out = out - M.Φ * M.pred
    end

    return out
end

function (M::AffineMap)(x)
    if islinear(M)
        return slope(M) * x
    end

    #out = zero(x)
    out = zeros(eltype(M), nout(M))

    if has_pred(M)
        out = out + slope(M) * (x - M.pred)
    else
        out = slope(M) * x
    end

    if has_prior(M)
        out = out + M.prior
    end

    return out
end

function compose(M2::AffineMap, M1::AffineMap)
    Φ = slope(M2) * slope(M1)
    prior = has_prior(M1) ? M2(M1.prior) : nothing
    pred = M1.pred

    return AffineMap(Φ, prior, pred)
end
