program test_csr_projections
  use sklearn_common, only: dp,require
  use sklearn_random_projection
  use sklearn_sparse, only: csr_matrix,csr_from_arrays,csr_from_dense
  implicit none
  type(gaussian_random_projection) :: gaussian,dense
  type(sparse_random_projection) :: sparse
  type(csr_matrix) :: x,v,large,query,projected
  real(dp),allocatable :: z(:,:),expected(:,:),back(:,:)
  x=csr_from_arrays(3,4,[1,4,4,6],[4,1,4,2,1],[2._dp,1._dp,-1._dp,3._dp,0._dp])
  gaussian%n_components=2;dense%n_components=2
  z=gaussian%fit_transform(x);expected=dense%fit_transform(x%to_dense())
  call require(all(gaussian%components_==dense%components_),'CSR fit preserves seeded components')
  call require(maxval(abs(z-expected))<1.e-14_dp,'CSR duplicates and empty rows')
  v=csr_from_dense(z);back=gaussian%inverse_transform(v);expected=dense%inverse_transform(z)
  call require(maxval(abs(back-expected))<1.e-13_dp,'CSR inverse coordinates')
  sparse%n_components=2;sparse%compute_inverse_components=.true.;call sparse%fit(x)
  z=sparse%transform(x);v=csr_from_dense(z);back=sparse%inverse_transform(v)
  projected=sparse%transform_sparse(x);call projected%validate();expected=projected%to_dense()
  call require(all(expected==z),'CSR output matches dense output')
  projected=sparse%fit_transform_sparse(x);expected=projected%to_dense()
  call require(all(expected==z),'CSR fit_transform output')
  expected=sparse%inverse_transform(z)
  call require(maxval(abs(back-expected))<1.e-13_dp,'CSR cached inverse')
  large%n_rows=100000;large%n_cols=10000
  allocate(large%indptr(large%n_rows+1),large%indices(0),large%data(0));large%indptr=1
  ! A dense copy would need eight GB. Fitting only uses CSR shape and validation.
  call gaussian%fit(large)
  sparse%compute_inverse_components=.false.;call sparse%fit(large)
  call require(gaussian%n_features_in_==10000.and.sparse%n_features_in_==10000,'large CSR shape-only fits')
  projected=sparse%transform_sparse(large)
  call require(projected%n_rows==100000.and.projected%nnz()==0,'large sparse output avoids dense sample matrix')
  query=csr_from_arrays(2,10000,[1,2,2],[10000],[1._dp]);z=gaussian%transform(query)
  call require(all(z(1,:)==gaussian%components_(:,10000)).and.all(z(2,:)==0),'large-width sparse query')
  print *, 'CSR projection fits, transforms, inverse maps, duplicates and large sparse shapes: PASS'
end program
