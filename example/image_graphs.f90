program image_graphs
  use sklearn, only: dp,csr_matrix,grid_to_graph,img_to_graph
  implicit none
  type(csr_matrix) :: lattice,gradient
  real(dp) :: image(2,3)
  real(dp), allocatable :: dense(:,:)
  logical :: mask(2,3)
  integer :: i
  lattice=grid_to_graph(4,5)
  print *, 'Grid nodes and stored entries:',lattice%n_rows,lattice%nnz()
  image(1,:)=[1._dp,2._dp,4._dp];image(2,:)=[8._dp,16._dp,32._dp]
  mask=.false.;mask(1,1)=.true.;mask(1,3)=.true.;mask(2,1)=.true.
  gradient=img_to_graph(image,mask);dense=gradient%to_dense()
  print *, 'Masked gradient nodes and stored entries:',gradient%n_rows,gradient%nnz()
  print *, 'Gradient adjacency (C-order selected pixels):'
  do i=1,size(dense,1)
    print '(3f8.2)',dense(i,:)
  end do
end program
