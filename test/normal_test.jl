


function normal_test(T,n)

    μ1 = randn(T,n)
    L1 = randn(T,n,n)
    Σ1 = Hermitian(L1*L1')

    x1 = randn(T,n)

    μ2 = randn(T,n)
    L2 = randn(T,n,n)
    Σ2 = Hermitian(L2*L2')

    N1 = Normal(μ1,Σ1)
    N2 = Normal(μ2,Σ2)

    if T<:Real

        logpdf1 = -1/2*logdet(2*π*Σ1) - 1/2*dot( x1-μ1, inv(Σ1), x1-μ1 )

        entropy1 = 1/2*logdet( 2*π*exp(1)*Σ1 )

        kld12 = 1/2*( tr( Σ2 \ Σ1 ) - n + dot(μ2 - μ1, inv(Σ2),μ2 - μ1 ) + logdet(Σ2) - logdet(Σ1)     )
        kld21 = 1/2*( tr( Σ1 \ Σ2 ) - n + dot(μ1 - μ2, inv(Σ1),μ1 - μ2 ) + logdet(Σ1) - logdet(Σ2)     )

    elseif T <:Complex

        logpdf1 = -n*log(π) - logdet(Σ1) - dot(x1-μ1,inv(Σ1),x1-μ1)

        entropy1 =  n*log(π) + logdet(Σ1) + n

        kld12 = real(tr(Σ2 \ Σ1)) - n  + real(dot(μ2-μ1,inv(Σ2),μ2-μ1))    +  logabsdet(Σ2)[1] - logabsdet(Σ1)[1]
        kld21 = real(tr(Σ1 \ Σ2)) - n  + real(dot(μ1-μ2,inv(Σ1),μ1-μ2))    +  logabsdet(Σ1)[1] - logabsdet(Σ2)[1]

    end

    @testset "Normal | $(T)" begin

    # correct values
    @test mean(N1) == μ1
    @test cov(N1) == Σ1
    @test var(N1) == diag(Σ1)
    @test std(N1) == sqrt.( diag(Σ1) )

    @test residual(N1,x1) ≈ cholesky(Σ1).L \ (x1 - μ1)
    @test logpdf(N1,x1) ≈ logpdf1

    @test entropy(N1) ≈ entropy1
    @test kldivergence(N1,N2) ≈ kld12
    @test kldivergence(N2,N1) ≈ kld21

    # correct type
    @test eltype(var(N1))<:Real
    @test eltype(std(N1))<:Real
    @test eltype(logpdf(N1,x1))<:Real
    @test eltype(entropy(N1))<:Real
    @test eltype(kldivergence(N1,N2))<:Real
    @test eltype(kldivergence(N2,N1))<:Real

    end


end
