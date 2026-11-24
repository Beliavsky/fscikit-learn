program multilabel_samples
  use sklearn, only: dp,make_multilabel_classification,mlp_classifier
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(mlp_classifier) :: model
  real(dp), allocatable :: x(:,:),prior(:),probability(:,:)
  integer, allocatable :: y(:,:)
  call make_multilabel_classification(80,12,x,y,n_classes=3,n_labels=1,length=15, &
    allow_unlabeled=.false.,random_state=42,p_c=prior)
  model%hidden_layer_sizes=[8];model%activation='tanh';model%max_iter=20
  model%batch_size=20;model%random_state=42;model%tol=0
  call model%fit(x,y);probability=model%predict_proba(x)
  print *, 'Samples, features and target columns:',size(x,1),size(x,2),size(y,2)
  print *, 'Every document has words and labels:',all(sum(x,dim=2)>0).and.all(sum(y,dim=2)>0)
  print *, 'Class prior sums to one:',abs(sum(prior)-1)<1.e-14_dp
  print *, 'Finite multilabel probabilities:',all(ieee_is_finite(probability))
  print *, 'Probability shape:',shape(probability)
end program
