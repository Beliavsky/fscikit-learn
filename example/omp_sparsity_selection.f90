program omp_sparsity_selection
  use sklearn, only: dp,orthogonal_matching_pursuit_cv
  implicit none
  type(orthogonal_matching_pursuit_cv) :: model
  real(dp) :: x(60,8),y(60)
  real(dp), allocatable :: prediction(:)
  integer :: i,j
  do i=1,60
    do j=1,8
      x(i,j)=sin(real(i*j,dp)*.31_dp)+cos(real(i*(j+1),dp)*.17_dp)
    end do
  end do
  y=1+2*x(:,1)-1.5_dp*x(:,3)+.8_dp*x(:,5)+.05_dp*sin([(real(i,dp),i=1,60)])
  model%cv=5; call model%fit(x,y); prediction=model%predict(x)
  print *, 'Selected active features: ',model%n_nonzero_coefs_
  print *, 'Mean validation MSE: ',model%best_mse_
  print *, 'Training R2: ',model%score(x,y)
  print *, 'Refit coefficients: ',model%coef_
end program
