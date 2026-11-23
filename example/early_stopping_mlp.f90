program early_stopping_mlp
  use sklearn, only: dp,mlp_regressor
  implicit none
  type(mlp_regressor) :: model
  real(dp) :: x(80,2),y(80)
  integer :: i
  do i=1,80
    x(i,:)=[sin(real(i,dp)),cos(.3_dp*i)]
    y(i)=.7_dp*x(i,1)-.2_dp*x(i,2)
  end do
  model%hidden_layer_sizes=[8];model%early_stopping=.true.
  model%validation_fraction=.25_dp;model%learning_rate_init=.02_dp
  model%max_iter=400;model%n_iter_no_change=12;model%tol=1.e-4_dp
  call model%fit(x,y)
  print *, 'Epochs and training samples seen:',model%n_iter_,model%t_
  print *, 'Training and validation partition sizes:',size(model%training_indices_),size(model%validation_indices_)
  print *, 'Best validation R2:',model%best_validation_score_
  print *, 'Restored validation R2:',model%score(x(model%validation_indices_,:),y(model%validation_indices_))
end program
