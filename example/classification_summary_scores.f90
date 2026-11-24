program classification_summary_scores
  use sklearn, only: dp,make_multilabel_classification,mlp_classifier,csr_matrix,csr_from_dense, &
    f1_score,fbeta_score,jaccard_score,precision_recall_fscore_support,multilabel_confusion_matrix
  implicit none
  type(mlp_classifier) :: model
  type(csr_matrix) :: truth_sparse,predicted_sparse
  real(dp), allocatable :: x(:,:),weights(:),precision(:),recall(:),fscore(:),support(:),cm(:,:,:)
  integer, allocatable :: y(:,:),predicted(:,:)
  integer :: i
  real(dp) :: value,dense_score
  call make_multilabel_classification(80,12,x,y,n_classes=3,n_labels=1,length=15, &
    allow_unlabeled=.false.,random_state=42)
  model%hidden_layer_sizes=[8];model%activation='tanh';model%max_iter=20
  model%batch_size=20;model%random_state=42;model%tol=0
  call model%fit(x,y);predicted=model%predict_indicator(x)
  weights=[(real(1+mod(i,3),dp),i=1,size(y,1))]
  dense_score=f1_score(y,predicted,average='macro',sample_weight=weights)
  print *, 'Weighted multilabel macro F1:',dense_score
  print *, 'Weighted sample Jaccard:',jaccard_score(y,predicted,average='samples',sample_weight=weights)
  call precision_recall_fscore_support(y,predicted,precision,recall,fscore,support,beta=2._dp,sample_weight=weights)
  print *, 'Per-label summary size:',size(precision)
  value=fbeta_score(y,predicted,2._dp,average='macro',sample_weight=weights)
  print *, 'F-beta summary agrees:',abs(value-sum(fscore)/size(fscore))<1.e-14_dp
  truth_sparse=csr_from_dense(real(y,dp));predicted_sparse=csr_from_dense(real(predicted,dp))
  value=f1_score(truth_sparse,predicted_sparse,average='macro',sample_weight=weights)
  print *, 'CSR F1 agrees:',abs(value-dense_score)<1.e-14_dp
  cm=multilabel_confusion_matrix(truth_sparse,predicted_sparse,sample_weight=weights)
  print *, 'Confusion tensor shape:',shape(cm)
end program
