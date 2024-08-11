module RecipesBaseExt

using RecipesBase
import MarkovKernels: Normal, AbstractNormal
using Statistics

for T in [:AbstractArray, :Matrix]
    @eval @recipe function f(y::$T{<:Normal}; ribbon_width = 1.96)
        means = mean.(y) |> stack
        stddevs = std.(y) |> stack
        ribbon --> ribbon_width * stddevs
        return means
    end

    @eval @recipe function f(x, y::$T{<:Normal}; ribbon_width = 1.96)
        means = mean.(y) |> stack
        stddevs = std.(y) |> stack
        ribbon --> ribbon_width * stddevs
        return x, means
    end

    # @eval @recipe function f(
    #     x::$T{<:Gaussian},
    #     y::$T{<:Gaussian};
    #     ribbon_width = 1.96,
    #     errors = true,
    # )
    #     xmeans = mean.(x) |> stack
    #     ymeans = mean.(y) |> stack
    #     if errors
    #         xstddevs = std.(x) |> stack
    #         xerror --> ribbon_width * xstddevs
    #         ystddevs = std.(y) |> stack
    #         yerror --> ribbon_width * ystddevs
    #     end
    #     return xmeans, ymeans
    # end

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
end

end
