program classification_example
  use sklearn
  implicit none
  type(standard_scaler) :: scaler
  type(logistic_regression) :: classifier
  real(dp) :: x(8,2)
  real(dp), allocatable :: scaled(:,:)
  integer :: y(8),i
  do i=1,8
    x(i,:)=[real(i,dp),real(mod(i,3),dp)]
  end do
  y=[0,0,0,0,1,1,1,1]
  call scaler%fit(x)
  scaled=scaler%transform(x)
  call classifier%fit(scaled,y)
  print *, 'Predictions: ',classifier%predict(scaled)
  print *, 'Training accuracy: ',classifier%score(scaled,y)
end program
