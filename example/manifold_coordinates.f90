program manifold_coordinates
  use sklearn
  implicit none
  type(mds) :: scaling
  type(locally_linear_embedding) :: lle
  real(dp) :: x(30,3),t
  real(dp), allocatable :: coordinates(:,:),projected(:,:)
  integer :: i
  do i=1,30
    t=real(i,dp)/10; x(i,:)=[sin(t),cos(t),0.2_dp*t+0.05_dp*sin(7*t)]
  end do
  scaling%random_state=42; scaling%eps=1.e-6_dp
  coordinates=scaling%fit_transform(x)
  print *, 'Metric MDS stress: ',scaling%stress_, ' iterations: ',scaling%n_iter_
  lle%n_neighbors=8; coordinates=lle%fit_transform(x); projected=lle%transform(x+0.01_dp)
  print *, 'LLE reconstruction objective: ',lle%reconstruction_error_
  print *, 'Projected query shape: ',shape(projected)
end program
