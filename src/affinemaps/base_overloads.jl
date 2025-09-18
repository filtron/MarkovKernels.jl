zero(a::AbstractAffineMap) =
    typeof(a)(Tuple(zero(getfield(a, f)) for f in fieldnames(typeof(a)))...)

function add_affinemaps(a1::AbstractAffineMap, a2::AbstractAffineMap)
    aout = AffineMap(slope(a1) + slope(a2), intercept(a1) + intercept(a2))
    return aout
end

+(a1::AbstractAffineMap, a2::AbstractAffineMap) = add_affinemaps(a1, a2)

-(a::LinearMap) = LinearMap(-slope(a))
-(a::AffineMap) = AffineMap(-slope(a), -intercept(a))
-(a::AffineCorrector) = AffineCorrector(-a.A, -a.b, a.c)
-(a1::AbstractAffineMap, a2::AbstractAffineMap) = +(a1, -(a2))

*(f, a::AbstractAffineMap) = AffineMap(f * slope(a), f * intercept(a))
*(f, a::LinearMap) = LinearMap(f * slope(a))
*(f, a::AffineCorrector) = AffineCorrector(f * a.A, f * a.b, a.c)

\(d, a::AbstractAffineMap) = AffineMap(d \ slope(a), d \ intercept(a))
\(d, a::LinearMap) = LinearMap(d \ slope(a))
\(d, a::AffineCorrector) = AffineCorrector(d \ a.A, d \ a.b, a.c)

copy(a::AbstractAffineMap) =
    typeof(a)(Tuple(copy(getfield(a, f)) for f in fieldnames(typeof(a)))...)

function copy!(Fdst::T, Fsrc::T) where {T<:AbstractAffineMap}
    fields = fieldnames(T)
    dst = Tuple(getfield(Fdst, f) for f in fields)
    src = Tuple(getfield(Fsrc, f) for f in fields)
    foreach(Base.splat(copy!), zip(dst, src))
    return Fdst
end

for func in (:(==), :isequal, :isapprox)
    @eval function Base.$func(a1::AbstractAffineMap, a2::AbstractAffineMap; kwargs...)
        T1, T2 = typeof(a1), typeof(a2)
        nameof(T1) === nameof(T2) || return false
        fields = fieldnames(T1)
        fields === fieldnames(T2) || return false

        for f in fields
            isdefined(a1, f) && isdefined(a2, f) || return false
            getfield(a1, f) === getfield(a2, f) ||
                $func(getfield(a1, f), getfield(a2, f); kwargs...) ||
                return false
        end
        return true
    end
end

# some fixes required for SISO/SIMO/MISO, sovle with Val{1} and Base.RefValue?
similar(a::LinearMap, ::Type{T}, dims::NTuple{2,IT}) where {T,IT} =
    LinearMap(similar(a.A, T, dims))
similar(a::AffineMap, ::Type{T}, dims::NTuple{2,IT}) where {T,IT} =
    AffineMap(similar(a.A, T, dims), similar(a.b, T, first(dims)))
similar(a::AffineCorrector, ::Type{T}, dims::NTuple{2,IT}) where {T,IT} = AffineCorrector(
    similar(a.A, T, dims),
    similar(a.b, T, first(dims)),
    similar(a.c, T, last(dims)),
)

similar(a::AbstractAffineMap, ::Type{T}, n, m) where {T} = similar(a, T, (n, m))
similar(a::AbstractAffineMap, ::Type{T}) where {T} = similar(a, T, size(slope(a)))
similar(a::AT) where {AT<:AbstractAffineMap} =
    AT(Tuple(similar(getfield(a, f)) for f in fieldnames(AT))...)

function similar(::Type{AT}, dims::NTuple{2,IDXT}) where {AT<:AbstractAffineMap,IDXT}
    n, m = dims
    ST = Base.promote_op(slope, AT)
    IT = Base.promote_op(intercept, AT)
    return AffineMap(similar(ST, n, m), similar(IT, n))
end

function similar(::Type{AT}, dims::NTuple{2,IDXT}) where {AT<:LinearMap,IDXT}
    n, m = dims
    ST = Base.promote_op(slope, AT)
    return LinearMap(similar(ST, n, m))
end

function similar(::Type{AT}, dims::NTuple{2,IDXT}) where {AT<:AffineCorrector,IDXT}
    n, m = dims
    TA, TB, TC = fieldtypes(AT)
    return AffineCorrector(similar(TA, n, m), similar(TB, n), similar(TC, m))
end

similar(::Type{AT}, n, m) where {AT<:AbstractAffineMap} = similar(AT, (n, m))
