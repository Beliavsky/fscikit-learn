program single_class_mlp
  use sklearn, only: dp,mlp_classifier,csr_matrix,csr_from_dense
  implicit none
  type(mlp_classifier) :: model
  type(csr_matrix) :: features
  real(dp) :: x(4,2)
  real(dp),allocatable :: probabilities(:,:)
  x=reshape([0._dp,1._dp,2._dp,3._dp,1._dp,0._dp,1._dp,0._dp],[4,2])
  features=csr_from_dense(x);model%hidden_layer_sizes=[integer::]
  call model%partial_fit(features,[7,7,7,7],classes=[7])
  call model%partial_fit(x,[7,7,7,7])
  probabilities=model%predict_proba(features)
  print *, 'Declared class:',model%classes_
  print *, 'Probability columns:',size(probabilities,2)
  print *, 'Predicted labels:',model%predict(features)
  print *, 'Training accuracy:',model%score(features,[7,7,7,7])
end program
