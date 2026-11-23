program test_mlp_partial
  use sklearn_neural_network, only: mlp_regressor,mlp_classifier
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(mlp_regressor) :: a,b,c,adam,whole,lb
  type(mlp_classifier) :: classifier,indicator
  real(dp) :: x(2,1),y(2),w,bias,velocity,bias_velocity,residual
  x(:,1)=[1._dp,2._dp];y=[1._dp,2._dp]
  a%hidden_layer_sizes=[integer::];a%solver='sgd';a%momentum=.7_dp;a%nesterovs_momentum=.false.
  a%batch_size=1;a%shuffle=.false.;a%learning_rate_init=.1_dp;a%alpha=.2_dp;a%max_iter=1;a%tol=0
  b=a;call a%partial_fit(x(:1,:),y(:1));c=a
  a%learning_rate_init=2;a%momentum=.1_dp;a%solver='adam'
  call a%partial_fit(x(2:,:),y(2:));call c%partial_fit(x(2:,:),y(2:));call b%fit(x,y)
  call require(all(a%layers_(1)%weights==b%layers_(1)%weights).and. &
    all(a%layers_(1)%bias==b%layers_(1)%bias),'SGD partial_fit retains velocity across batches')
  call require(all(a%layers_(1)%weights==c%layers_(1)%weights).and.a%optimizer_kind_=='sgd', &
    'partial_fit preserves cached optimizer parameters and type')
  call require(a%n_iter_==1.and.a%t_==2.and.size(a%loss_curve_)==2,'partial_fit one-epoch history and clock')
  adam%hidden_layer_sizes=[integer::];adam%batch_size=1;adam%shuffle=.false.;adam%max_iter=1;adam%tol=0
  adam%beta_1=.6_dp;adam%beta_2=.83_dp;adam%epsilon=.003_dp;adam%learning_rate_init=.03_dp
  whole=adam;call adam%partial_fit(x(:1,:),y(:1));call adam%partial_fit(x(2:,:),y(2:));call whole%fit(x,y)
  call require(all(adam%layers_(1)%weights==whole%layers_(1)%weights).and. &
    all(adam%layers_(1)%m==whole%layers_(1)%m),'Adam partial_fit retains moments and batch timestep')
  call require(adam%optimizer_step_==2,'Adam incremental bias-correction clock')
  adam%warm_start=.true.;adam%batch_size=0;adam%beta_1=.4_dp;call adam%fit(x,y)
  call require(adam%optimizer_step_==1.and.adam%optimizer_beta1_==.4_dp,'ordinary warm fit recreates optimizer')
  call classifier%partial_fit(x(:1,:),[3],classes=[11,3,7,7])
  call classifier%partial_fit(x(2:,:),[11])
  call require(all(classifier%classes_==[3,7,11]).and.classifier%n_outputs_==3,'declared-class incremental OVR output')
  classifier%t_=2147483647_int64;call classifier%partial_fit(x(:1,:),[7])
  call require(classifier%t_==2147483648_int64,'incremental 64-bit sample count')
  call indicator%partial_fit(x(:1,:),reshape([1,0,1],[1,3]),classes=[0,1,2])
  call require(indicator%multilabel_.and.indicator%n_outputs_==3,'incremental multilabel targets')
  lb%solver='lbfgs';lb%hidden_layer_sizes=[integer::];lb%max_iter=1
  call lb%partial_fit(x,y)
  call require(lb%t_==0.and.lb%n_iter_==1.and.allocated(lb%gradient_), &
    'sklearn 1.6.1 regression partial_fit L-BFGS behavior')
  c%solver='lbfgs';c%warm_start=.false.;c%max_iter=1;call c%fit(x,y)
  call require(allocated(c%loss_curve_).and.c%optimizer_kind_=='sgd','cold L-BFGS preserves existing stochastic state')
  w=c%layers_(1)%weights(1,1);bias=c%layers_(1)%bias(1)
  velocity=c%layers_(1)%m(1,1);bias_velocity=c%layers_(1)%mb(1)
  residual=2*w+bias-2
  velocity=.7_dp*velocity-.1_dp*(2*residual+.2_dp*w);bias_velocity=.7_dp*bias_velocity-.1_dp*residual
  c%solver='adam';call c%partial_fit(x(2:,:),y(2:))
  call require(abs(c%layers_(1)%weights(1,1)-(w+velocity))<1.e-14_dp.and. &
    abs(c%layers_(1)%bias(1)-(bias+bias_velocity))<1.e-14_dp,'cached SGD survives cold L-BFGS and resumes')
  print *, 'MLP partial fits, declared classes and persistent optimizers: PASS'
end program
