program csr_projections
  use sklearn, only: dp,gaussian_random_projection,sparse_random_projection
  use sklearn_sparse, only: csr_matrix,csr_from_arrays,csr_from_dense
  implicit none
  type(gaussian_random_projection) :: gaussian
  type(sparse_random_projection) :: sparse
  type(csr_matrix) :: x,projected,latent
  real(dp),allocatable :: z(:,:),back(:,:)
  x=csr_from_arrays(3,4,[1,4,4,6],[4,1,4,2,1],[2._dp,1._dp,-1._dp,3._dp,0._dp])
  gaussian%n_components=2
  z=gaussian%fit_transform(x);latent=csr_from_dense(z)
  back=gaussian%inverse_transform(latent)
  print *, 'Gaussian first row:',z(1,:)
  print *, 'Inverse first row:',back(1,:)
  sparse%n_components=2;sparse%density=.5_dp
  projected=sparse%fit_transform_sparse(x)
  print *, 'Sparse output shape:',projected%n_rows,projected%n_cols
  print *, 'Sparse row pointers:',projected%indptr
  print *, 'Sparse column indices:',projected%indices
  print *, 'Sparse stored values:',projected%data
end program
