program sparse_latent_semantics
  use sklearn, only: dp,truncated_svd
  use sklearn_sparse, only: csr_matrix,csr_from_arrays
  implicit none
  type(truncated_svd) :: model
  type(csr_matrix) :: documents
  real(dp), allocatable :: latent(:,:)
  documents=csr_from_arrays(6,8,[1,4,7,10,13,16,19], &
    [1,2,4,1,3,4,2,3,4,5,6,8,5,7,8,6,7,8], &
    [2._dp,1._dp,1._dp,1._dp,2._dp,1._dp,1._dp,1._dp,2._dp,2._dp,1._dp,1._dp, &
     1._dp,2._dp,1._dp,1._dp,1._dp,2._dp])
  model%algorithm='randomized';model%n_components=2;model%random_state=42
  latent=model%fit_transform(documents)
  print '(a,2i5)', 'Latent document shape: ',shape(latent)
  print '(a,2f14.8)', 'Singular values: ',model%singular_values_
  print '(a,2f14.8)', 'Explained variance ratios: ',model%explained_variance_ratio_
end program
