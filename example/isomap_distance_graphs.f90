program isomap_distance_graphs
  use sklearn, only: dp,isomap,csr_matrix,csr_from_dense,pairwise_distances
  implicit none
  type(isomap) :: radius,precomputed
  type(csr_matrix) :: graph,query
  real(dp) :: x(8,2)
  real(dp), allocatable :: d(:,:),z(:,:),projected(:,:)
  x(:,1)=[0._dp,1._dp,2._dp,3._dp,10._dp,11._dp,12._dp,13._dp]
  x(:,2)=[0._dp,0._dp,.2_dp,.4_dp,2._dp,2._dp,2._dp,2._dp]
  radius%n_neighbors=0;radius%radius=1.5_dp;radius%path_method='D'
  z=radius%fit_transform(x)
  print *, 'Original radius components:',radius%n_connected_components_
  print *, 'Embedding dimensions:',shape(z)
  d=pairwise_distances(x,x);graph=csr_from_dense(d,.true.)
  precomputed%metric='precomputed';precomputed%n_neighbors=4;precomputed%path_method='D'
  call precomputed%fit(graph)
  query=csr_from_dense(d(:2,:),.true.);projected=precomputed%transform(query)
  print *, 'CSR query dimensions:',shape(projected)
  print *, 'Reconstruction error:',precomputed%reconstruction_error()
end program
