program coordinate_descent_nmf
  use sklearn, only: dp,nmf
  implicit none
  type(nmf) :: model
  real(dp) :: x(4,3),initial_w(4,1),initial_h(1,3)
  real(dp), allocatable :: weights(:,:)
  integer :: i
  do i=1,4
    x(i,:)=real(i,dp)*[1._dp,2._dp,3._dp]
  end do
  model%solver='cd';model%init='custom';model%n_components=1
  initial_w=1;initial_h(1,:)=[1._dp,2._dp,3._dp]
  weights=model%fit_transform(x,initial_w,initial_h)
  print '(a,i0)', 'Coordinate-descent iterations: ',model%n_iter_
  print '(a,es14.6)', 'Reconstruction error: ',model%reconstruction_err_
  print '(a,3f12.6)', 'Learned component: ',model%components_(1,:)
end program
