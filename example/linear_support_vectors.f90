program linear_support_vectors
  use sklearn, only: dp,linear_svc,linear_svr
  implicit none
  type(linear_svc) :: classifier
  type(linear_svr) :: regressor
  real(dp) :: x(4,1)
  x(:,1)=[-2._dp,-1._dp,1._dp,2._dp]
  call classifier%fit(x,[-3,-3,9,9])
  call regressor%fit(x,[-4._dp,-2._dp,2._dp,4._dp])
  print *, 'Classification accuracy:',classifier%score(x,[-3,-3,9,9])
  print *, 'Regression predictions:',regressor%predict(x)
end program
