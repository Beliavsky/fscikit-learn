program gaussian_process_sampling
  use sklearn, only: dp,gaussian_process_regressor,rbf_covariance,white_covariance,operator(+)
  implicit none
  type(gaussian_process_regressor) :: model
  real(dp) :: x(5,1),query(3,1)
  real(dp), allocatable :: samples(:,:)
  x(:,1)=[-1._dp,-.5_dp,0._dp,.5_dp,1._dp];query(:,1)=[-.8_dp,.1_dp,.7_dp]
  model%kernel=rbf_covariance(.7_dp)+white_covariance(.02_dp)
  model%alpha=.04_dp;model%normalize_y=.true.;call model%fit(x,sin(x(:,1)))
  samples=model%sample_y(query,n_samples=4,random_state=42)
  print '(a,3f13.8)', 'Posterior mean: ',model%predict(query)
  print '(a,12f13.8)', 'Four native draws: ',samples
end program
