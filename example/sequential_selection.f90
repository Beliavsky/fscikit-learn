program sequential_selection
  use sklearn, only: dp,linear_regression,sequential_feature_selector
  implicit none
  type(linear_regression) :: estimator
  type(sequential_feature_selector) :: selector
  real(dp) :: x(40,4),y(40)
  real(dp), allocatable :: reduced(:,:)
  integer :: i
  do i=1,40
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),sin(real(2*i,dp)),cos(real(3*i,dp))]
  end do
  y=3*x(:,2)-2*x(:,4)
  selector%n_features_to_select=2
  reduced=selector%fit_transform(x,y,estimator)
  print *, 'Selected feature mask: ',selector%get_support()
  call estimator%fit(reduced,y)
  print *, 'Selected-feature R2: ',estimator%score(reduced,y)
end program
