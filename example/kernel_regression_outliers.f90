program kernel_regression_outliers
  use sklearn
  implicit none
  type(svr) :: regression
  type(nu_svr) :: adaptive
  type(one_class_svm) :: novelty
  real(dp) :: x(40,2),y(40),t,query(2,2)
  integer, allocatable :: labels(:)
  integer :: i
  do i=1,40
    t=real(i,dp)/10; x(i,:)=[sin(t),cos(1.7_dp*t)]; y(i)=sin(2*t)+0.12_dp*sin(17*t)
  end do
  regression%c=2; regression%gamma=0.7_dp; call regression%fit(x,y)
  print *, 'SVR training R2: ',regression%score(x,y)
  adaptive%c=2; adaptive%gamma=0.7_dp; call adaptive%fit(x,y)
  print *, 'NuSVR learned epsilon: ',adaptive%epsilon_
  novelty%nu=0.2_dp; call novelty%fit(x)
  query(1,:)=x(20,:); query(2,:)=10; labels=novelty%predict(query)
  print *, 'Training-near and distant query labels: ',labels
end program
