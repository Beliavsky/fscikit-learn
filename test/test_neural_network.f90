program test_neural_network
  use sklearn_common
  use sklearn_neural_network
  implicit none
  type(mlp_classifier) :: classifier
  type(mlp_regressor) :: regressor
  real(dp) :: xor(4,2),x(30,1),y(30)
  real(dp), allocatable :: prob(:,:)
  integer :: i
  xor(:,1)=[0._dp,0._dp,1._dp,1._dp]; xor(:,2)=[0._dp,1._dp,0._dp,1._dp]
  classifier%hidden_layer_sizes=[8,4]; classifier%activation='tanh'
  classifier%max_iter=2500; classifier%learning_rate_init=0.02_dp
  call classifier%fit(xor,[0,1,1,0])
  call require(classifier%score(xor,[0,1,1,0])==1,'MLP learns nonlinear XOR')
  prob=classifier%predict_proba(xor)
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-12_dp,'MLP normalized probabilities')
  call require(classifier%loss_curve_(classifier%n_iter_)<classifier%loss_curve_(1),'MLP classification loss')
  do i=1,30
    x(i,1)=real(i-15,dp)/15; y(i)=sin(2*x(i,1))
  end do
  regressor%hidden_layer_sizes=[12]; regressor%activation='tanh'
  regressor%learning_rate_init=0.02_dp; regressor%max_iter=2000
  regressor%tol=1.e-6_dp; regressor%n_iter_no_change=20
  call regressor%fit(x,y); call require(regressor%score(x,y)>0.99_dp,'MLP nonlinear regression')
  print *, 'neural networks: PASS'
end program
