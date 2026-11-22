program streaming_document_nb
  use sklearn_common, only: dp
  use sklearn_naive_bayes, only: multinomial_nb
  implicit none
  type(multinomial_nb) :: model
  real(dp) :: documents(4,3),query(2,3)
  real(dp), allocatable :: probability(:,:)
  integer :: labels(4)
  documents=reshape([3._dp,2._dp,0._dp,0._dp,1._dp,0._dp,2._dp,3._dp,0._dp,1._dp,3._dp,2._dp],[4,3])
  query=reshape([2._dp,0._dp,1._dp,2._dp,0._dp,3._dp],[2,3]);labels=[0,0,1,1]
  call model%partial_fit(documents(:2,:),labels(:2),[0,1],[1._dp,2._dp])
  call model%partial_fit(documents(3:,:),labels(3:),sample_weight=[.5_dp,1._dp])
  probability=model%predict_proba(query)
  print '(a,2f14.8)', 'Weighted class counts: ',model%class_count_
  print '(a,2f14.8)', 'First query probabilities: ',probability(1,:)
  print '(a,2i5)', 'Predicted classes: ',model%predict(query)
end program
