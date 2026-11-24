program seeded_projections
  use sklearn, only: dp,gaussian_random_projection,sparse_random_projection
  implicit none
  type(gaussian_random_projection) :: gaussian
  type(sparse_random_projection) :: sparse
  real(dp) :: x(2,3)
  real(dp),allocatable :: z(:,:)
  x=reshape([1._dp,2._dp,3._dp,4._dp,5._dp,6._dp],[2,3])
  gaussian%n_components=2;gaussian%random_state=0
  z=gaussian%fit_transform(x)
  print *, 'Gaussian first component:',gaussian%components_(1,:)
  print *, 'Gaussian first row:',z(1,:)
  sparse%n_components=2;sparse%random_state=0;sparse%density=.5_dp
  z=sparse%fit_transform(x)
  print *, 'Sparse first component:',sparse%components_(1,:)
  print *, 'Sparse first row:',z(1,:)
end program
