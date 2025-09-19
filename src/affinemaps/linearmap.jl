"""
    LinearMap{T,U}

Type for representing affine maps with zero intercept.
"""
struct LinearMap{T,U} <: AbstractAffineMap{T}
    A::U
end

"""
    LinearMap(A::AbstractMatrix)

Creates a LinearMap with slope A.
"""
LinearMap(A) = LinearMap{eltype(A),typeof(A)}(A)

Base.iterate(a::LinearMap) = (a.A, Val(:done))
Base.iterate(::LinearMap, ::Val{:done}) = nothing

slope(a::LinearMap) = a.A
intercept(a::LinearMap{T}) where {T} = zeros(T, size(slope(a), 1))
intercept(::LinearMap{T,<:Adjoint}) where {T} = zero(T)
intercept(::LinearMap{T,<:Number}) where {T} = zero(T)

(a::LinearMap)(x) = slope(a) * x

function (a::LinearMap)(y, x)
    mul!(y, slope(a), x)
    return y
end

compose(a2::LinearMap, a1::LinearMap) = LinearMap(slope(a2) * slope(a1))

LinearMap{T}(a::LinearMap) where {T} = LinearMap(convert(AbstractMatrix{T}, a.A))
LinearMap{T}(a::LinearMap{<:Number,<:Number}) where {T} = LinearMap(convert(T, a.A))
AbstractAffineMap{T}(a::LinearMap) where {T} = LinearMap{T}(a)

function Base.show(io::IO, a::LinearMap{T,U}) where {T,U}
    print(io, summary(a))
    print(io, "\n A = ")
    show(io, (a.A))
end
