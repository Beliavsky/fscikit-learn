program streaming_sparse_features
  use sklearn, only: dp,mini_batch_dictionary_learning,mini_batch_sparse_pca
  implicit none
  type(mini_batch_dictionary_learning) :: dictionary
  type(mini_batch_sparse_pca) :: projection
  real(dp) :: x(12,4)
  real(dp), allocatable :: code(:,:),scores(:,:),reconstructed(:,:)
  integer :: i
  do i=1,12
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),sin(.3_dp*i),cos(.7_dp*i)]
  end do
  dictionary%n_components=3; dictionary%fit_algorithm='cd'; dictionary%transform_algorithm='lasso_cd'
  dictionary%alpha=.1_dp; dictionary%batch_size=4; dictionary%random_state=42
  call dictionary%partial_fit(x(:6,:)); call dictionary%partial_fit(x(7:,:))
  code=dictionary%transform(x)
  print *, 'Streaming dictionary updates: ',dictionary%n_steps_
  print *, 'Sparse code nonzeros: ',count(abs(code)>1.e-12_dp)
  projection%n_components=2; projection%max_iter=20; projection%alpha=.1_dp; projection%random_state=42
  scores=projection%fit_transform(x); reconstructed=projection%inverse_transform(scores)
  print *, 'Sparse PCA epochs/updates: ',projection%n_iter_,projection%n_steps_
  print *, 'Sparse PCA reconstruction RMSE: ',sqrt(sum((x-reconstructed)**2)/size(x))
end program
