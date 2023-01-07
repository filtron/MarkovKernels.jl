@recipe function f(
    ::Type{T},
    ds::T,
) where {T<:Dirac{<:Number,<:AbstractVector{<:AbstractVector{<:Number}}}}
    values = mapreduce(permutedims, vcat, mean(ds))

    return values
end

@recipe function f(
    ::Type{T},
    ps::T,
) where {T<:AbstractVector{<:ParticleSystem{<:Number,<:AbstractVector}}}
    if !allequal(dim.(ps))
        error("all elements of ns must be of the same dimension")
    end

    dimension = dim(first(ps))
    X = mapreduce(permutedims, vcat, particles.(ps))

    #seriestype := :scatter
    layout --> (dimension, 1)
    linestyle --> :dot
    alpha --> 0.1
    color --> "black"

    xs = [getindex.(X, i) for i in 1:dimension]

    for (i, x) in enumerate(xs)
        @series begin
            subplot := i
            label --> string(i)
            x
        end
    end

    primary := false
    ()
end
