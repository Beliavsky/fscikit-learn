program gaussian_process_kernels
  use sklearn, only: dp,gp_kernel,matern_covariance,white_covariance,operator(*),operator(+), &
    gaussian_process_regressor,gaussian_process_classifier
  implicit none
  type(gp_kernel) :: kernel
  type(gaussian_process_regressor) :: reg
  type(gaussian_process_classifier) :: classifier
  real(dp) :: x(20,1),y(20),q(4,1)
  real(dp), allocatable :: probability(:,:)
  integer :: i,labels(20)
  do i=1,20
    x(i,1)=4*real(i-1,dp)/19
  end do
  y=sin(x(:,1));labels=merge(1,0,x(:,1)>2);q(:,1)=[.3_dp,1.7_dp,3._dp,4.8_dp]
  kernel=1.4_dp*matern_covariance(.8_dp,nu=1.5_dp)+white_covariance(.03_dp)
  reg%kernel=kernel;reg%alpha=.01_dp;reg%normalize_y=.true.;call reg%fit(x,y)
  print '(a,4f12.7)', 'GP mean: ',reg%predict(q)
  print '(a,4f12.7)', 'GP std:  ',reg%predict_std(q)
  classifier%kernel=kernel;call classifier%fit(x,labels);probability=classifier%predict_proba(q)
  print '(a,4f12.7)', 'GP positive-class probability: ',probability(:,2)
end program
