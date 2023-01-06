@recipe function f(
    ::Type{T},
    ns::T,
) where {T<:Dirac{<:Number,<:AbstractVector{<:AbstractVector{<:Number}}}}
    values = mapreduce(permutedims, vcat, mean(ns))

    return values
end


@recipe function f(
    ::Type{T},
    ns::T,
) where {T<:ParticleSystem{<:Number,<:AbstractVector}}

    values = mapreduce(permutedims, vcat, particles.(ns))

    return values
end
