program test_mlp_stochastic
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(mlp_regressor) :: r
  type(mlp_classifier) :: c
  real(dp) :: x(2,1),saved
  real(dp), allocatable :: probability(:,:),logs(:,:)
  x(:,1)=[1._dp,2._dp]
  r%hidden_layer_sizes=[integer::];r%solver='sgd';r%momentum=0;r%nesterovs_momentum=.false.
  r%batch_size=1;r%shuffle=.false.;r%learning_rate_init=.1_dp;r%alpha=.2_dp;r%max_iter=1;r%tol=0
  call r%fit(x,[1._dp,2._dp])
  ! Seeded two-batch SGD, independently checked against sklearn 1.6.1.
  call require(abs(r%layers_(1)%weights(1,1)-.35027375108310477_dp)<1.e-14_dp,'MLP SGD batch coefficients')
  call require(abs(r%layers_(1)%bias(1)-.8437341467837521_dp)<1.e-14_dp,'MLP SGD batch intercept')
  call require(abs(r%loss_-.20614887212229624_dp)<1.e-14_dp,'MLP weighted batch loss')
  call require(r%t_==2.and.r%n_features_in_==1.and.r%n_outputs_==1.and.r%n_layers_==2,'MLP fit metadata')
  saved=r%layers_(1)%weights(1,1);call r%fit(x,[1._dp,2._dp])
  call require(r%layers_(1)%weights(1,1)==saved.and.r%t_==2,'MLP refit resets parameters and optimizer')
  c%hidden_layer_sizes=[integer::];c%solver='sgd';c%learning_rate='adaptive'
  c%tol=2;c%n_iter_no_change=1;c%max_iter=65
  call c%fit(x,[0,1])
  call require(c%n_iter_==13.and.c%converged_,'MLP adaptive reductions and strict patience stopping')
  call require(c%out_activation_=='logistic'.and.size(c%layers_(1)%bias)==1,'MLP single-output binary network')
  c%layers_(1)%weights=1;c%layers_(1)%bias=0
  probability=c%predict_proba(reshape([0._dp,1000._dp],[2,1]))
  logs=c%predict_log_proba(reshape([0._dp,1000._dp],[2,1]))
  call require(all(probability(1,:)==.5_dp).and.probability(2,2)==1,'MLP binary probability conversion')
  call require(.not.ieee_is_finite(logs(2,1)).and.logs(2,2)==0,'MLP saturated log probabilities')
  call require(all(c%predict(reshape([0._dp,1000._dp],[2,1]))==[0,1]),'MLP binary threshold ties')
  probability=c%predict_proba(reshape([-710._dp,-709._dp],[2,1]))
  logs=c%predict_log_proba(reshape([-710._dp,-709._dp],[2,1]))
  call require(probability(1,2)==0.and.probability(2,2)>0,'MLP source expit negative saturation boundary')
  call require(.not.ieee_is_finite(logs(1,2)).and.ieee_is_finite(logs(2,2)),'MLP source log probability boundary')
  print *, 'Stochastic MLP batches, schedules and binary outputs: PASS'
end program
