program pca_solver_selection
  use sklearn, only: dp,pca
  implicit none
  type(pca) :: model
  real(dp) :: x(520,60)
  real(dp), allocatable :: z(:,:)
  integer :: i,j
  do j=1,60
    do i=1,520
      x(i,j)=sin(.21_dp*i)*cos(.31_dp*j)+cos(.13_dp*i)*sin(.43_dp*j)+.2_dp*sin(real(i*j,dp))
    end do
  end do
  model%n_components=3;model%svd_solver='auto';model%random_state=42;z=model%fit_transform(x)
  print '(a,a)', 'Selected solver: ',trim(model%fit_svd_solver_)
  print '(a,3f14.9)', 'Explained variance: ',model%explained_variance_
  print '(a,2i5)', 'Training shape: ',shape(z)
end program
