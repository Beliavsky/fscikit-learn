program test_gpr_optimizer
  use sklearn_gaussian_process, only: gaussian_process_regressor
  use sklearn_gp_kernels
  use sklearn_common, only: dp,require
  implicit none
  type(gaussian_process_regressor) :: model,fixed
  type(gp_kernel) :: a,b,k,copy
  real(dp) :: x(20,1),y(20),bounds(3,2),value
  real(dp), allocatable :: theta(:),combined(:,:)
  integer :: i
  a=rbf_covariance();a=a%with_bounds(reshape(log([.2_dp,2._dp]),[1,2]))
  b=white_covariance(.03_dp);b=b%with_bounds(reshape(log([.001_dp,.5_dp]),[1,2]))
  k=(a+b)**2;combined=k%bounds()
  call require(all(combined(1,:)==log([.2_dp,2._dp])).and.all(combined(2,:)==log([.001_dp,.5_dp])), &
    'GP composition and powers retain child bounds')
  copy=k%with_theta(k%theta()+.1_dp)
  call require(all(copy%bounds()==combined),'GP theta clones retain bounds')
  do i=1,20
    x(i,1)=4*real(i-1,dp)/19
  end do
  y=sin(x(:,1))+.03_dp*cos(9*x(:,1));bounds(:,1)=log([.1_dp,.1_dp,.0001_dp]);bounds(:,2)=log([4._dp,4._dp,.4_dp])
  k=1.3_dp*rbf_covariance(.9_dp)+white_covariance(.04_dp);model%kernel=k%with_bounds(bounds)
  model%alpha=.02_dp;model%normalize_y=.true.;fixed=model;call fixed%fit(x,y)
  model%optimizer='fmin_l_bfgs_b';model%n_restarts_optimizer=1;model%random_state=42;call model%fit(x,y)
  call require(model%log_marginal_likelihood_>=fixed%log_marginal_likelihood_-1.e-8_dp,'GPR optimization improves likelihood')
  call require(model%n_optimization_runs_==2.and.model%n_function_evaluations_>0,'GPR optimizer and restart diagnostics')
  call require(model%converged_,'GPR optimizer convergence')
  theta=model%kernel_%theta()
  call require(all(theta>=bounds(:,1)).and.all(theta<=bounds(:,2)),'GPR optimization respects log bounds')
  value=model%log_marginal_likelihood(theta)
  call require(abs(value-model%log_marginal_likelihood_)<1.e-10_dp,'GPR optimized likelihood cache')
  print *, 'GPR bounded optimization and restarts: PASS'
end program
