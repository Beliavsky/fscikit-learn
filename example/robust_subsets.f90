program robust_subsets
  use sklearn
  implicit none
  type(theil_sen_regressor) :: robust
  type(linear_regression) :: ordinary
  real(dp) :: x(25,1),y(25)
  integer :: i
  x(:,1)=[(real(i,dp)/10,i=1,25)]; y=2*x(:,1)+0.5_dp
  y(:3)=y(:3)+12
  robust%tol=1.e-8_dp; robust%max_iter=1000
  call robust%fit(x,y); call ordinary%fit(x,y)
  print *, 'Theil-Sen slope/intercept:',robust%coef_,robust%intercept_
  print *, 'Ordinary slope/intercept:',ordinary%coef_,ordinary%intercept_
  print *, 'Subset fits:',robust%n_subpopulation_,'converged:',robust%converged_
end program
