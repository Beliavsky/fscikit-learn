program seeded_rbf_features
  use sklearn,only: dp,rbf_sampler
  use sklearn_sparse,only: csr_matrix,csr_from_arrays
  implicit none
  type(rbf_sampler)::model
  type(csr_matrix)::x
  real(dp),allocatable::z(:,:)
  x=csr_from_arrays(3,3,[1,3,3,4],[1,3,2],[1._dp,2._dp,3._dp])
  model%n_components=4;model%random_state=42;model%gamma_scale=.true.
  z=model%fit_transform(x)
  print *, 'Scaled gamma:',model%gamma_
  print *, 'First weight row:',model%random_weights_(1,:)
  print *, 'Uniform offsets:',model%random_offset_
  print *, 'First feature row:',z(1,:)
end program
