program gaussian_process_regression
  use sklearn, only: dp,gaussian_process_regressor,rbf_covariance,white_covariance,operator(*),operator(+),gp_kernel
  implicit none
  type(gaussian_process_regressor) :: model
  type(gp_kernel) :: kernel
  real(dp) :: x(24,1),y(24,2),query(3,1),bounds(3,2)
  real(dp), allocatable :: mean(:,:),std(:,:)
  integer :: i
  do i=1,24
    x(i,1)=4*real(i-1,dp)/23
  end do
  y(:,1)=sin(x(:,1));y(:,2)=cos(x(:,1));query(:,1)=[.5_dp,2._dp,3.5_dp]
  kernel=1.2_dp*rbf_covariance(.8_dp)+white_covariance(.04_dp)
  bounds(:,1)=log([.1_dp,.1_dp,.0001_dp]);bounds(:,2)=log([5._dp,4._dp,.5_dp])
  model%kernel=kernel%with_bounds(bounds);model%alpha_values=.01_dp+.01_dp*x(:,1)
  model%normalize_y=.true.;model%optimizer='fmin_l_bfgs_b';model%n_restarts_optimizer=1;model%random_state=42
  call model%fit(x,y);mean=model%predict_multi(query);std=model%predict_std_multi(query)
  print '(a,3f12.7)', 'First target mean:  ',mean(:,1)
  print '(a,3f12.7)', 'Second target mean: ',mean(:,2)
  print '(a,3f12.7)', 'First target std:   ',std(:,1)
  print '(a,f14.7)', 'Joint likelihood:  ',model%log_marginal_likelihood_
end program
