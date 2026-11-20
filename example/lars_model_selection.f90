program lars_model_selection
  use sklearn, only: dp,lasso_lars_cv,lasso_lars_ic
  implicit none
  type(lasso_lars_cv) :: cv_model
  type(lasso_lars_ic) :: bic_model
  real(dp) :: x(48,5),y(48)
  integer :: i
  do i=1,48
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),sin(real(2*i,dp)),cos(real(3*i,dp)),sin(real(4*i,dp))]
    y(i)=2+3*x(i,2)-2*x(i,4)+.2_dp*cos(.7_dp*i)
  end do
  cv_model%cv=4; call cv_model%fit(x,y)
  bic_model%criterion='bic'; call bic_model%fit(x,y)
  print *, 'CV alpha and coefficients: ',cv_model%alpha_,cv_model%coef_
  print *, 'BIC alpha and coefficients: ',bic_model%alpha_,bic_model%coef_
  print *, 'Estimated noise variance: ',bic_model%noise_variance_
  print *, 'CV and BIC R2: ',cv_model%score(x,y),bic_model%score(x,y)
end program
