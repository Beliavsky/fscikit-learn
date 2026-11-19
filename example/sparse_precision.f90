program sparse_precision
  use sklearn
  implicit none
  type(graphical_lasso) :: model
  real(dp) :: covariance(3,3)
  integer :: j
  covariance=reshape([1._dp,0.6_dp,0.1_dp,0.6_dp,1._dp,0.4_dp,0.1_dp,0.4_dp,1._dp],[3,3])
  model%covariance='precomputed'; model%alpha=0.2_dp; model%tol=1.e-8_dp; model%enet_tol=1.e-10_dp
  call model%fit(covariance)
  print *, 'Sparse precision matrix:'
  do j=1,3
    print '(3f12.6)',model%precision_(j,:)
  end do
  print *, 'Converged:',model%converged_,'iterations:',model%n_iter_,'dual gap:',model%dual_gap_
end program
