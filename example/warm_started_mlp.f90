program warm_started_mlp
  use sklearn_neural_network, only: mlp_classifier
  use sklearn_common, only: dp
  implicit none
  type(mlp_classifier) :: model
  real(dp) :: x(4,2)
  real(dp), allocatable :: probabilities(:,:)
  integer :: i
  x(:,1)=[0._dp,0._dp,1._dp,1._dp];x(:,2)=[0._dp,1._dp,0._dp,1._dp]
  model%hidden_layer_sizes=[8,4];model%activation='tanh';model%solver='adam'
  model%learning_rate_init=.02_dp;model%batch_size=2;model%max_iter=100
  model%tol=1.e-6_dp;model%n_iter_no_change=20;model%warm_start=.true.
  do i=1,5
    call model%fit(x,[0,1,1,0])
  end do
  probabilities=model%predict_proba(x)
  print *, 'Predicted XOR labels:',model%predict(x)
  print *, 'Last call epochs and total history:',model%n_iter_,size(model%loss_curve_)
  print *, 'Total samples seen:',model%t_
  print *, 'Final training loss:',model%loss_
  print *, 'Probability row sums:',sum(probabilities,dim=2)
end program
