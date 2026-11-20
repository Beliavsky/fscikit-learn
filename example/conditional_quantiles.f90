program conditional_quantiles
  use sklearn, only: dp,quantile_regressor
  implicit none
  type(quantile_regressor) :: model
  real(dp) :: x(40,1),y(40),quantiles(3)
  integer :: i,k
  do i=1,40
    x(i,1)=real(i,dp)/10
    y(i)=1+2*x(i,1)+(.2_dp+x(i,1)/5)*sin(real(i*7,dp))
  end do
  model%alpha=.01_dp; quantiles=[.1_dp,.5_dp,.9_dp]
  do k=1,3
    model%quantile=quantiles(k); call model%fit(x,y)
    print *, 'quantile, slope, intercept:',model%quantile,model%coef_,model%intercept_
    print *, 'converged, primal-dual gap:',model%converged_,model%dual_gap_
  end do
end program
