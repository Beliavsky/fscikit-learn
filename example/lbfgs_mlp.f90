program lbfgs_mlp
  use sklearn_neural_network, only: mlp_classifier
  use sklearn_common, only: dp
  implicit none
  type(mlp_classifier) :: model
  real(dp) :: x(4,2)
  real(dp), allocatable :: probabilities(:,:)
  x(:,1)=[0._dp,0._dp,1._dp,1._dp];x(:,2)=[0._dp,1._dp,0._dp,1._dp]
  model%hidden_layer_sizes=[8,4];model%activation='tanh';model%solver='lbfgs'
  model%max_iter=200;model%max_fun=15000;model%tol=1.e-6_dp
  call model%fit(x,[0,1,1,0])
  probabilities=model%predict_proba(x)
  print *, 'Predicted XOR labels:',model%predict(x)
  print *, 'Optimizer iterations and evaluations:',model%n_iter_,model%n_function_evaluations_
  print *, 'Final training loss:',model%loss_
  print *, 'Probability row sums:',sum(probabilities,dim=2)
end program
