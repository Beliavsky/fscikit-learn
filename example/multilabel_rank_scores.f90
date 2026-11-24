program multilabel_rank_scores
  use sklearn, only: dp,make_multilabel_classification,mlp_classifier,csr_matrix,csr_from_dense, &
    coverage_error,label_ranking_average_precision_score,label_ranking_loss
  implicit none
  type(mlp_classifier) :: model
  type(csr_matrix) :: indicators
  integer, allocatable :: y(:,:)
  real(dp), allocatable :: x(:,:),probability(:,:)
  real(dp) :: precision
  call make_multilabel_classification(80,12,x,y,n_classes=3,n_labels=1,length=15, &
    allow_unlabeled=.false.,random_state=42)
  model%hidden_layer_sizes=[8];model%activation='tanh';model%max_iter=20
  model%batch_size=20;model%random_state=42;model%tol=0
  call model%fit(x,y);probability=model%predict_proba(x)
  precision=label_ranking_average_precision_score(y,probability)
  indicators=csr_from_dense(real(y,dp))
  print *, 'Multilabel score shape:',shape(probability)
  print *, 'Coverage error:',coverage_error(y,probability)
  print *, 'Ranking average precision:',precision
  print *, 'Ranking loss:',label_ranking_loss(y,probability)
  print *, 'CSR precision agrees:',abs(precision-label_ranking_average_precision_score(indicators,probability))<1.e-14_dp
  print *, 'Perfect indicator ranking loss:',label_ranking_loss(y,real(y,dp))
end program
