const AbstractNumOrVec{T} = Union{T,AbstractVector{T}} where {T<:Number}
const Scalar{T} = Union{T,Base.RefValue{T}} where {T<:Number}
