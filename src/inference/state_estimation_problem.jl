
abstract type AbstractStateEstimationProblem end

# maybe subtype AbstractStateEstimationProblem wither other abstract type for problems that have length /

struct HomogeneousStateEstimationProblem{M,I,FK,MK,B} <: AbstractStateEstimationProblem
    measurements::M
    init::I
    forward_kernel::FK
    measurement_kernel::MK
    aligned::B
    function HomogeneousStateEstimationProblem(ys::AbstractVecOrMat,init::AbstractDistribution,fw_kernel::AbstractMarkovKernel,m_kernel::AbstractMarkovKernel,aligned::Bool)
        new{typeof(ys),typeof(init),typeof(fw_kernel),typeof(m_kernel),typeof(aligned)}(ys,init,fw_kernel,m_kernel,aligned)
    end
end

isaligned(problem::HomogeneousStateEstimationProblem) = problem.aligned
initial_distribution(problem::HomogeneousStateEstimationProblem) = problem.init

Base.IteratorSize(problem::HomogeneousStateEstimationProblem) = Base.HasLength()
length(problem::HomogeneousStateEstimationProblem) = isaligned(problem) ? size(problem.measurements,1) : size(problem.measurements,1)+1 # i.e. number of filter steps in total



function make_likelihood(measurement_kernel,y)
    likelihood = Likelihood(measurement_kernel,y)
    return likelihood
end

function make_likelihood(measurement_kernel::AbstractNormalKernel,y::AbstractVector)

    # build projection  matrix
    idx = findall(!isnan,y) # assumign missing measurements are encoded by NaN
    if isempty(idx)
        return nothing
    else
        P = zeros(eltype(y),length(idx),length(y))
        P[1:length(idx),idx] = Matrix( one(eltype(y))*I, length(idx),length(idx))
    end

    projection_kernel = DiracKernel(P)

    projected_measurement_kernel = compose( projection_kernel, measurement_kernel )
    projected_y = y[idx]

    return Likelihood(projected_measurement_kernel,projected_y)

end

function Base.iterate(problem::HomogeneousStateEstimationProblem, state=1)

    if length(problem) < state
        return nothing
    end

    if state == 1

        if isaligned(problem)
            fw_kernel = nothing
          #  likelihood = Likelihood( problem.measurement_kernel, problem.measurements[state,:])
            likelihood = make_likelihood(problem.measurement_kernel,problem.measurements[state,:])
        else
            fw_kernel = problem.forward_kernel
            likelihood = nothing
        end

    else

        fw_kernel = problem.forward_kernel
        #likelihood = Likelihood( problem.measurement_kernel, problem.measurements[state,:])
        likelihood = make_likelihood(problem.measurement_kernel,problem.measurements[state,:])

    end

    out = (fw_kernel,likelihood)

    return out, state+1

end


