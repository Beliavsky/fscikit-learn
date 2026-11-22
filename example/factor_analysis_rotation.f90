program factor_analysis_rotation
  use sklearn, only: dp,factor_analysis
  implicit none
  type(factor_analysis) :: model
  real(dp) :: x(60,8)
  real(dp), allocatable :: factors(:,:)
  integer :: i,j
  do j=1,8
    do i=1,60
      x(i,j)=sin(.2_dp*i)*cos(.3_dp*j)+cos(.13_dp*i)*sin(.5_dp*j)+.2_dp*sin(real(i*j,dp))
    end do
  end do
  model%n_components=2;model%rotation='varimax';model%svd_method='randomized'
  model%iterated_power=3;model%random_state=42;factors=model%fit_transform(x)
  print '(a,2i5)', 'Latent shape: ',shape(factors)
  print '(a,f14.8)', 'Mean log likelihood: ',model%score(x)
  print '(a,2i5)', 'Fit and rotation iterations: ',model%n_iter_,model%n_rotation_iter_
end program
