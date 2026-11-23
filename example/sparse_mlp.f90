program sparse_mlp
  use sklearn, only: dp,mlp_classifier,csr_matrix,csr_from_dense
  implicit none
  type(mlp_classifier) :: model
  type(csr_matrix) :: features
  real(dp) :: x(6,4)
  integer :: labels(6),i
  x=0;labels=[0,0,0,1,1,1]
  do i=1,6
    x(i,1)=real(i,dp)-3.5_dp
  end do
  features=csr_from_dense(x)
  model%hidden_layer_sizes=[integer::];model%solver='sgd'
  model%learning_rate_init=.1_dp;model%batch_size=2;model%shuffle=.false.
  do i=1,100
    call model%partial_fit(features,labels,classes=[0,1])
  end do
  print *, 'Stored values and dense feature count:',features%nnz(),size(x)
  print *, 'Predicted labels:',model%predict(features)
  print *, 'Training accuracy:',model%score(features,labels)
end program
