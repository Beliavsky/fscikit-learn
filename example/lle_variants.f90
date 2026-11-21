program lle_variants
  use sklearn, only: dp,locally_linear_embedding
  implicit none
  type(locally_linear_embedding) :: model
  real(dp) :: x(24,4)
  real(dp), allocatable :: embedding(:,:),query(:,:)
  integer :: i,j,method
  do i=1,24
    do j=1,4
      x(i,j)=sin(.731_dp*i*j)+cos(.213_dp*i*i+.191_dp*j)+.02_dp*i*j
    end do
  end do
  model%n_neighbors=11; model%n_components=2
  do method=1,2
    model%method='hessian'
    if(method==2) model%method='modified'
    embedding=model%fit_transform(x); query=model%transform(x(:3,:)+.017_dp)
    print *, trim(model%method),' embedding shape: ',shape(embedding)
    print *, 'Alignment reconstruction energy: ',model%reconstruction_error_
    print *, 'First query coordinates: ',query(1,:)
  end do
end program
