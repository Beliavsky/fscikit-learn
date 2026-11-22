program probabilistic_pca
  use sklearn, only: dp,pca
  implicit none
  type(pca) :: model
  real(dp) :: x(120,8)
  real(dp), allocatable :: z(:,:)
  integer :: i,j
  do j=1,8
    do i=1,120
      x(i,j)=sin(.21_dp*i)*cos(.31_dp*j)+cos(.13_dp*i)*sin(.43_dp*j)+.2_dp*sin(real(i*j,dp))
    end do
  end do
  model%dimension_mode='mle';z=model%fit_transform(x)
  print '(a,i5)', 'Selected Minka dimension: ',model%n_components_
  print '(a,f14.8)', 'Noise variance: ',model%noise_variance_
  print '(a,f14.8)', 'Mean log likelihood: ',model%score(x)
  model%dimension_mode='variance';model%variance_fraction=.9_dp;call model%fit(x)
  print '(a,i5)', 'Dimension for over 90 percent variance: ',model%n_components_
end program
