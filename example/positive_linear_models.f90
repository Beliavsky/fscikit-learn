program positive_linear_models
  use sklearn, only: dp,linear_regression,ridge
  implicit none
  type(linear_regression) :: model
  type(ridge) :: regularized
  real(dp) :: x(4,2),y(4)
  x(:,1)=[-1._dp,1._dp,-1._dp,1._dp];x(:,2)=[-1._dp,-1._dp,1._dp,1._dp]
  y=2-x(:,1)+3*x(:,2)
  model%positive=.true.;call model%fit(x,y)
  regularized%positive=.true.;regularized%alpha=4;call regularized%fit(x,y)
  print '(a,2f12.6)', 'Nonnegative OLS coefficients: ',model%coef_
  print '(a,f12.6)', 'Unconstrained intercept: ',model%intercept_
  print '(a,2f12.6)', 'Nonnegative Ridge coefficients: ',regularized%coef_
end program
