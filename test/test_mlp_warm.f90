program test_mlp_warm
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(mlp_regressor) :: r,again,l
  type(mlp_classifier) :: c
  real(dp) :: x(2,1),y(2),w,b,residual
  real(dp), allocatable :: history(:)
  integer :: i
  x(:,1)=[1._dp,2._dp];y=[1._dp,2._dp]
  r%hidden_layer_sizes=[integer::];r%solver='sgd';r%momentum=0;r%nesterovs_momentum=.false.
  r%batch_size=1;r%shuffle=.false.;r%learning_rate_init=.1_dp;r%alpha=.2_dp;r%max_iter=1;r%tol=0
  call r%fit(x,y)
  w=r%layers_(1)%weights(1,1);b=r%layers_(1)%bias(1)
  do i=1,2
    residual=x(i,1)*w+b-y(i)
    w=w-.1_dp*(x(i,1)*residual+.2_dp*w);b=b-.1_dp*residual
  end do
  r%warm_start=.true.;r%random_state=7;call r%fit(x,y)
  call require(abs(r%layers_(1)%weights(1,1)-w)<1.e-14_dp.and.abs(r%layers_(1)%bias(1)-b)<1.e-14_dp, &
    'warm MLP preserves coefficients rather than reinitializing')
  call require(r%n_iter_==1.and.r%t_==4.and.size(r%loss_curve_)==2,'warm MLP history and call/sample clocks')
  again=r;r%solver='adam';again%solver='adam'
  again%layers_(1)%m=100;again%layers_(1)%v=100;again%layers_(1)%mb=100;again%layers_(1)%vb=100
  call r%fit(x,y);call again%fit(x,y)
  call require(all(r%layers_(1)%weights==again%layers_(1)%weights).and. &
    all(r%layers_(1)%bias==again%layers_(1)%bias),'warm MLP reconstructs Adam optimizer state')
  r%warm_start=.false.;call r%fit(x,y)
  call require(r%t_==2.and.size(r%loss_curve_)==1,'cold MLP resets history and sample count')
  history=r%loss_curve_;r%warm_start=.true.;r%solver='lbfgs';r%tol=1000
  call r%fit(x,y)
  call require(r%n_iter_==0.and.r%t_==2.and.all(r%loss_curve_==history),'warm L-BFGS preserves stochastic history')
  r%solver='sgd';r%tol=0;r%t_=2147483640_int64;r%max_iter=2;r%learning_rate='invscaling'
  call r%fit(x,y)
  call require(r%t_==2147483644_int64,'warm MLP uses a 64-bit sample clock')
  r%max_iter=3;call r%fit(x,y)
  call require(r%t_==2147483650_int64,'warm MLP clock crosses signed 32-bit boundary')
  l%hidden_layer_sizes=[integer::];l%solver='lbfgs';l%tol=1000
  call l%fit(x,y);l%layers_(1)%weights=.5_dp;l%warm_start=.true.;call l%fit(x,y)
  call require(l%layers_(1)%weights(1,1)==.5_dp.and.l%n_iter_==0,'warm L-BFGS respects published parameters')
  c%hidden_layer_sizes=[3];c%max_iter=1;call c%fit(x,[3,7])
  c%warm_start=.true.;c%hidden_layer_sizes=[7];call c%fit(x,[3,7])
  call require(size(c%layers_(1)%bias)==3,'stochastic warm fits retain actual layer widths')
  c%layers_(1)%weights=spread(spread(.1_dp,1,1),2,7);c%layers_(1)%bias=spread(.05_dp,1,7)
  c%layers_(2)%weights=spread(spread(-.1_dp,1,7),2,1);call c%fit(x,[3,7])
  call require(size(c%layers_(1)%m,2)==7.and.size(c%layers_(2)%m,1)==7,'warm optimizer follows edited parameter shapes')
  c%warm_start=.false.;call c%fit(x,[3,7])
  call require(size(c%layers_(1)%bias)==7.and.size(c%loss_curve_)==1,'cold fit applies new architecture')
  print *, 'MLP warm fits, histories and optimizer resets: PASS'
end program
