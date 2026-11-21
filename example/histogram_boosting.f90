program histogram_boosting
  use sklearn, only: dp,hist_gradient_boosting_regressor,hist_gradient_boosting_classifier
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(hist_gradient_boosting_regressor) :: regressor
  type(hist_gradient_boosting_classifier) :: classifier
  real(dp) :: x(120,2),y(120)
  integer :: j,labels(120)
  do j=1,120
    x(j,1)=real(j,dp)/120;x(j,2)=sin(real(j,dp));y(j)=2*x(j,1)+x(j,2)**2
    labels(j)=merge(0,1,j<=60)
  end do
  x(15,1)=ieee_value(0._dp,ieee_quiet_nan)
  regressor%max_iter=30;regressor%min_samples_leaf=5
  classifier%max_iter=15;classifier%min_samples_leaf=5
  call regressor%fit(x,y);call classifier%fit(x,labels)
  print '(a,f10.6)', 'Histogram regressor training R2: ',regressor%score(x,y)
  print '(a,f10.6)', 'Histogram classifier training accuracy: ',classifier%score(x,labels)
end program
