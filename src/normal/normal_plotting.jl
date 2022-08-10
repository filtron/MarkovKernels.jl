@recipe function f(::Type{T}, ns::T) where {T<:AbstractVector{<:AbstractNormal}}
    ribbon_width = 1.96
    dimensions = dim(ns[1])

    if !all(dim.(ns) .== dimensions)
        error("all elements of ns must be of the same dimension")
    end

    N = length(ns)
    times = 0:N-1

    values = mapreduce(permutedims, vcat, mean.(ns))
    stds = mapreduce(permutedims, vcat, std.(ns))

    #display(values)

    ribbon --> ribbon_width * stds
    xguide --> "t"
    yguide --> "y"

    return values
end
