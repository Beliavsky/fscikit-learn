program sparse_document_nb
  use sklearn, only: dp,csr_matrix,csr_from_arrays,multinomial_nb
  implicit none
  type(csr_matrix) :: documents,query
  type(multinomial_nb) :: model
  real(dp), allocatable :: probability(:,:)
  documents=csr_from_arrays(4,10000,[1,3,5,7,9],[1,7,1,20,7,10000,7,10000], &
    [3._dp,1._dp,2._dp,1._dp,2._dp,3._dp,3._dp,2._dp])
  call model%fit(documents,[0,0,1,1],sample_weight=[1._dp,2._dp,.5_dp,1._dp])
  query=csr_from_arrays(2,10000,[1,3,5],[1,7,7,10000],[2._dp,1._dp,2._dp,3._dp])
  probability=model%predict_proba(query)
  print '(a,i0)', 'Sparse feature width: ',model%n_features_in_
  print '(a,2f14.8)', 'First document probabilities: ',probability(1,:)
  print '(a,2i4)', 'Predicted classes: ',model%predict(query)
end program
