program gaussian_process_classification
  use sklearn, only: dp,gaussian_process_classifier,gp_kernel,rbf_covariance,white_covariance,operator(*),operator(+)
  implicit none
  type(gaussian_process_classifier) :: model
  type(gp_kernel) :: kernel
  real(dp) :: x(24,1),query(3,1),bounds(3,2)
  real(dp), allocatable :: probability(:,:)
  integer :: i,labels(24)
  do i=1,24
    x(i,1)=-1.5_dp+3*real(i-1,dp)/23
    labels(i)=merge(-4,merge(9,17,i<=16),i<=8)
  end do
  kernel=1.4_dp*rbf_covariance(.7_dp)+white_covariance(.04_dp)
  bounds(:,1)=log([.1_dp,.1_dp,.0001_dp]);bounds(:,2)=log([4._dp,4._dp,.4_dp])
  model%kernel=kernel%with_bounds(bounds);model%optimizer='fmin_l_bfgs_b'
  model%n_restarts_optimizer=1;model%random_state=42;call model%fit(x,labels)
  query(:,1)=[-1.2_dp,0._dp,1.2_dp];probability=model%predict_proba(query)
  print '(a,3i5)', 'Predicted classes: ',model%predict(query)
  print '(a,3f12.7)', 'Middle-class probability: ',probability(:,2)
  print '(a,f14.7)', 'Mean marginal likelihood: ',model%log_marginal_likelihood()
end program
