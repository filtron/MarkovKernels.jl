@recipe function f(::Type{T}, ns::T) where {T<:AbstractVector{<:AbstractNormal}}
    ribbon_width = 1.96

    if !allequal(dim.(ns))
        error("all elements of ns must be of the same dimension")
    end

    values = mapreduce(permutedims, vcat, mean.(ns))
    stds = mapreduce(permutedims, vcat, std.(ns))

    ribbon --> ribbon_width * stds
    return values
end
