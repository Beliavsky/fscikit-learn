program shrunk_discriminant_projection
  use sklearn_common, only: dp
  use sklearn_discriminant_analysis, only: linear_discriminant_analysis
  implicit none
  type(linear_discriminant_analysis) :: model
  real(dp) :: x(30,4)
  integer :: labels(30),i,j
  real(dp), allocatable :: projected(:,:)
  do i=1,30
    labels(i)=mod(i,3)
    do j=1,3
      x(i,j)=sin(.4_dp*i*j)+labels(i)*.7_dp*j
    end do
  end do
  x(:,4)=x(:,1)
  model%solver='eigen';model%shrinkage=.2_dp;model%n_components=2
  projected=model%fit_transform(x,labels)
  print '(a,2i5)', 'Projected shape: ',shape(projected)
  print '(a,2f14.8)', 'Explained variance ratios: ',model%explained_variance_ratio_
  print '(a,f14.8)', 'Training accuracy: ',model%score(x,labels)
end program
