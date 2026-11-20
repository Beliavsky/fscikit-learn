program least_angle_regression
  use sklearn, only: dp,lasso_lars
  implicit none
  type(lasso_lars) :: model
  real(dp) :: x(40,4),y(40)
  integer :: i
  do i=1,40
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),sin(real(2*i,dp)),cos(real(3*i,dp))]
  end do
  y=2+3*x(:,2)-2*x(:,4); model%alpha=.05_dp
  call model%fit(x,y)
  print *, 'Coefficients: ',model%coef_
  print *, 'Intercept: ',model%intercept_
  print *, 'Alpha knots: ',model%alphas_
  print *, 'R2: ',model%score(x,y)
end program
