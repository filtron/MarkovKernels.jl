function covariance_parameter_test(T)
    n  = 1 
    m = 2 

    Vp = tril(ones(T, m, m)) 
    Vp = Vp*Vp'
    Cp = ones(T, n, m) 
    Rp = ones(T, n, n) 

    matrix_containers = (:Matrix, :SMatrix)

    # Diagonal is a bit broken for StaticArrays, I admit defeat. 
    #cov_wrappers = (:AbstractMatrix, :Diagonal)
    cov_wrappers = (:AbstractMatrix,)
    cov_types = (:HermOrSym, :Cholesky)

    for con in matrix_containers, covw in cov_wrappers, covpt in cov_types 

        V = _make_covp(_wrap_matrix(_make_matrix(Vp, con), covw), covpt)
        C = _make_matrix(Cp, con)
        R = _make_covp(_wrap_matrix(_make_matrix(Rp, con), covw), covpt) 

        @testset "stein | $(con) | $(covw) | $(covpt)" begin 
            @test _ofsametype(_make_matrix(Rp, con), stein(V, C))
            @test _ofsametype(_make_matrix(Rp, con), stein(V, C, R))
        end

        S1, K1, Σ1 = schur_reduce(V, C)
        S2, K2, Σ2 = schur_reduce(V, C, R)

        @testset "schur_reduce | $(con) | $(covw) | $(covpt)" begin 
            @test _ofsametype(_make_matrix(Rp, con), S1) 
            @test _ofsametype(permutedims(C), K1) 
            @test _ofsametype(_make_matrix(Vp, con), Σ1)

            @test _ofsametype(_make_matrix(Rp, con), S2) 
            @test _ofsametype(permutedims(C), K2) 
            @test _ofsametype(_make_matrix(Vp, con), Σ2)
        end

    end

end

function _make_matrix(A::AbstractMatrix, s) 
    n, m  = size(A) 
    if s === :Matrix 
        return Matrix(A)
    elseif s === :SMatrix 
        return SMatrix{n,m}(A)
    end
end

function _wrap_matrix(A::AbstractMatrix, s)  
    if s === :AbstractMatrix 
        return A 
    elseif s === :Diagonal 
        return Diagonal(A) 
    end  
end

function _make_covp(A::AbstractMatrix{T}, s) where {T}
    if s === :HermOrSym 
        return T <: Complex ? Hermitian(A) : Symmetric(A) 
    elseif s === :Cholesky 
        return cholesky(A) 
    end
end

function _ofsametype(Ain::AbstractMatrix, Aout::AbstractMatrix)  
    typeof(Aout) <: typeof(Ain)
end

function _ofsametype(Ain::AbstractMatrix, Aout::HermOrSym)  
    typeof(parent(Aout)) <: typeof(Ain)
end

function _ofsametype(Ain::Diagonal, Aout::HermOrSym)  
    typeof(parent(Aout)) <: typeof(diagm(parent(Ain)))
end

function _ofsametype(Ain::AbstractMatrix, Aout::Cholesky)  
    typeof(Aout.factors) <: typeof(Ain)
end