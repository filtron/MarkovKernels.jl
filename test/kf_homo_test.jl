


function kf_homo_test()


    N = 2^8+1

    a = sqrt(1/3)
    b = sqrt(1/2)
    Φ = [2*a -a^2-b^2; 1.0 0]
    Q =  Matrix(1.0*I,2,2)
    C = [1.0 0]
    R = fill(0.1,1,1)

    m0 = zeros(2)
    V0 = Matrix(1.0*I,2,2)

    init = Normal(m0,V0)
    forward_kernel = NormalKernel(Φ,Q)
    output_kernel = DiracKernel(C)
    measurement_kernel = NormalKernel(C,R)


    xs = rand(init,forward_kernel,N-1)
    output = rand(output_kernel,xs)
    ys = rand(measurement_kernel,xs)

    fs1, ps1, bs1, ll1 = bayes_filter(ys,init,forward_kernel,measurement_kernel,true)

    problem = HomogeneousStateEstimationProblem(ys,init,forward_kernel,measurement_kernel,true)

    fs2, ps2, bs2, ll2 = bayes_filter(problem)

    @testset "homogeneous Kalman filter | " begin

        @test length(fs1) == length(fs2)
        @test length(ps1) == length(ps2)
        @test length(bs1) == length(bs2)
        @test ll1 == ll2

    end



end

function _kf_for_testing(ys,m0,V0,Φ,Q,C,R,aligned)


    dimx = length(m0)
    N, dimy = size(ys)
    loglike = 0.0

    filter_mean = m0
    filter_cov = V0

    # N-1 steps
    if aligned
        filter_means = zeros(N,dimx)
        filter_covs = zeros(dimx,dimx,N)
        filter_preds = zeros(N,dimx)
        smoother_gains = zeros(dimx,dimx,N-1)

        pred_means = zeros(N,dimy)
        pred_covs = zeros(dimy,dimy,N)
    # N steps
    else
        filter_means = zeros(N,dimx)
        filter_covs = zeros(dimx,dimx,N)
        filter_preds = zeros(N,dimx)
        smoother_gains = zeros(dimx,dimx,N-1)

        pred_means = zeros(N,dimy)
        pred_covs = zeros(dimy,dimy,N)

        filter_means[1,:] = filter_mean
        filter_covs[:,:,1] = filter_cov
    end



end


function _invert_for_testing(y,μ,Π,C,R)

    K = Π*C'
    S = Hermitian(C*K + R)
    K = K/S

    L = (I - K*C)

    pred_mean = C*μ
    pred_cov = Matrix(S)

    m = μ + K*(y-pred_mean)
    Σ = Matrix(Hermitian(L*Π*L' + K*R*K'))

    return pred_mean, pred_cov, m, Σ

end