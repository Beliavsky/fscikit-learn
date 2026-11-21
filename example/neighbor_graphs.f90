! Distance graphs keep stored zero edges, which dense arrays cannot distinguish
! from missing edges. Connectivity graphs can aggregate neighboring values.
program neighbor_graphs
  use sklearn, only: dp,kneighbors_transformer,radius_neighbors_transformer,csr_matrix
  implicit none
  type(kneighbors_transformer) :: nearest
  type(radius_neighbors_transformer) :: radius
  type(csr_matrix) :: graph
  real(dp) :: x(5,1),values(5,1)
  real(dp), allocatable :: totals(:,:)
  x(:,1)=[0._dp,1._dp,2._dp,4._dp,8._dp]; values(:,1)=[1._dp,2._dp,3._dp,4._dp,5._dp]
  nearest%n_neighbors=2
  graph=nearest%fit_transform(x)
  print '(a,i0)', 'Stored distance edges (k+1 per row): ',graph%nnz()
  radius%radius=1; radius%mode='connectivity'
  graph=radius%fit_transform(x); totals=graph%matmul(values)
  print '(a,*(f6.1))', 'Radius-one neighbor value sums: ',totals(:,1)
end program
