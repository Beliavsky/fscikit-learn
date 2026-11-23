program beta_divergence_nmf
  use sklearn, only: dp,nmf
  implicit none
  type(nmf) :: model
  real(dp) :: x(5,4)
  real(dp), allocatable :: weights(:,:),codes(:,:)
  integer :: i
  do i=1,5
    x(i,:)=[real(i,dp),1._dp,real(6-i,dp),2._dp]
  end do
  model%solver='mu';model%beta_loss=1;model%init='random';model%random_state=42
  model%n_components=2;model%max_iter=300;model%tol=1.e-5_dp
  model%alpha_w=.01_dp;model%l1_ratio=.5_dp
  weights=model%fit_transform(x);codes=model%transform(x)
  print '(a,i0)', 'KL NMF iterations: ',model%n_iter_
  print '(a,es24.16)', 'Square-root beta divergence: ',model%reconstruction_err_
  print '(a,es24.16)', 'Fixed-component code sum: ',sum(codes)
end program
