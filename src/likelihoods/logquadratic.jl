"""
    LogQuadraticLikelihood

Type for representing log-quadratic likelihoods, i.e. 
    
log L(x) = c - 0.5 * norm(y - C * x)^2 

"""
struct LogQuadraticLikelihood{A,B,C} <: AbstractLikelihood
    logconst::A
    y::B
    C::C
end

Base.iterate(L::LogQuadraticLikelihood) = (L.logconst, Val(:y))
Base.iterate(L::LogQuadraticLikelihood, ::Val{:y}) = (L.y, Val(:C))
Base.iterate(L::LogQuadraticLikelihood, ::Val{:C}) = (L.C, Val(:done))
Base.iterate(::LogQuadraticLikelihood, ::Val{:done}) = nothing

# consider letting C be an affine (linear?) map instead 
function LogQuadraticLikelihood(L::Likelihood{<:AffineHomoskedasticNormalKernel})
    K, y = measurement_model(L), measurement(L)
    F = mean(K)
    Rsqrt = lsqrt(covp(K))
    m = length(y)

    ybar = Rsqrt \ (y - intercept(F))
    Cbar = Rsqrt \ slope(F)
    logc = -m / 2 * log(2π) - logdet(Rsqrt)
    return LogQuadraticLikelihood(logc, ybar, Cbar)
end

logconstant(L::LogQuadraticLikelihood) = L.logconst

observation(L::LogQuadraticLikelihood) = L.y

observation_matrix(L::LogQuadraticLikelihood) = L.C

function log(L::LogQuadraticLikelihood, x)
    logc, y, C = L
    return logc - norm(y - C * x)^2 / 2
end
