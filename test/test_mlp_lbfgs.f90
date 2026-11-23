program test_mlp_lbfgs
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_common, only: dp,require
  implicit none
  type(mlp_regressor) :: r,again
  type(mlp_classifier) :: c
  real(dp) :: x(5,2),y(5),z(6,2)
  real(dp), allocatable :: probability(:,:)
  x(:,1)=[-2._dp,-1._dp,0._dp,1._dp,2._dp];x(:,2)=[0._dp,1._dp,0._dp,1._dp,0._dp]
  y=1+2*x(:,1)-.5_dp*x(:,2)
  r%hidden_layer_sizes=[integer::];r%solver='lbfgs';r%alpha=.2_dp;r%tol=1.e-10_dp
  call r%fit(x,y)
  ! No hidden layer reduces squared-error MLP to ridge with an unpenalized intercept.
  call require(maxval(abs(r%layers_(1)%weights(:,1)-[20._dp/10.2_dp,-.6_dp/1.4_dp]))<1.e-5_dp, &
    'MLP L-BFGS solves ridge normal equations')
  call require(abs(r%layers_(1)%bias(1)-(.8_dp+.4_dp*.6_dp/1.4_dp))<1.e-5_dp,'MLP unpenalized intercept')
  call require(maxval(abs(r%gradient_))<1.e-5_dp.and.r%converged_,'MLP L-BFGS final gradient and convergence')
  call require(r%t_==0.and..not.allocated(r%loss_curve_),'L-BFGS has no stochastic sample clock or loss curve')
  again=r;again%shuffle=.false.;again%batch_size=1;again%learning_rate_init=2
  call again%fit(x,y)
  call require(all(again%layers_(1)%weights==r%layers_(1)%weights),'L-BFGS ignores stochastic controls')
  r%max_iter=1;call r%fit(x,y)
  call require(r%n_iter_==1.and..not.r%converged_,'MLP L-BFGS iteration limit')
  r%max_iter=200;r%max_fun=1;call r%fit(x,y)
  call require(r%n_function_evaluations_>1.and.index(r%optimization_message_,'EVALUATIONS')>0, &
    'MLP L-BFGS soft function-evaluation limit')
  r%max_fun=15000;r%tol=1000;call r%fit(x,y)
  call require(r%n_iter_==0.and.r%n_function_evaluations_==1.and.r%converged_, &
    'MLP L-BFGS can stop at initialization')
  r%solver='adam';r%tol=0;r%max_iter=2;call r%fit(x,y)
  call require(allocated(r%loss_curve_).and..not.allocated(r%gradient_).and.r%t_==10, &
    'MLP solver changes clear stale optimizer diagnostics')
  c%hidden_layer_sizes=[integer::];c%solver='lbfgs';c%alpha=.01_dp;c%tol=1.e-8_dp
  z(:,1)=[-2._dp,-1._dp,2._dp,1._dp,0._dp,0._dp];z(:,2)=[0._dp,0._dp,0._dp,0._dp,2._dp,1._dp]
  call c%fit(z,[3,3,7,7,11,11]);probability=c%predict_proba(z)
  call require(all(c%predict(z)==[3,3,7,7,11,11]),'MLP L-BFGS multiclass softmax')
  call require(all(abs(sum(probability,dim=2)-1)<1.e-14_dp),'MLP L-BFGS normalized probabilities')
  print *, 'MLP L-BFGS objectives, limits and solver changes: PASS'
end program
