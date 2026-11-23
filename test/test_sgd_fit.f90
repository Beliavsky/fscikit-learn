program test_sgd_fit
  use sklearn, only: dp,sgd_regressor,sgd_classifier
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(sgd_regressor) :: r
  type(sgd_classifier) :: c
  real(dp) :: x(1,1),y(1),z(2,1),initial(1),bias
  real(dp), allocatable :: values(:,:)
  character(32) :: losses(4)=[character(32)::'squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']
  integer :: i
  x=1;y=2;r%penalty='none';r%learning_rate='constant';r%eta0=.1_dp;r%max_iter=1;r%tol=-1
  r%warm_start=.true.;call r%fit(x,y);call r%fit(x,y,coef_init=[99._dp],intercept_init=9._dp)
  call require(abs(r%coef_(1)-.36_dp)<1.e-14_dp.and.abs(r%intercept_-.36_dp)<1.e-14_dp, &
    'regressor warm fit reuses iterates and source ignores new initializers')
  call require(r%t_==2,'warm fit resets the sample clock')
  r%warm_start=.false.;initial=1;bias=.5_dp;call r%fit(x,y,coef_init=initial,intercept_init=bias)
  call require(abs(r%coef_(1)-1.05_dp)<1.e-14_dp.and.abs(r%intercept_-.55_dp)<1.e-14_dp,'explicit regression seed')
  call require(all(initial==1).and.bias==.5_dp,'initializers are preserved')
  r%average=1;call r%fit(x,y);r%warm_start=.true.;call r%fit(x,y)
  call require(abs(r%coef_(1)-.56_dp)<1.e-14_dp,'source regressor retains accumulated average across warm fits')
  z(:,1)=[-1._dp,1._dp];c%penalty='none';c%learning_rate='constant';c%eta0=.1_dp
  c%max_iter=1;c%tol=-1;c%average=1;c%warm_start=.true.
  call c%fit(z,[1,2]);call c%fit(z,[1,2])
  call require(abs(c%coef_(1,1)-.3_dp)<1.e-14_dp.and.abs(c%intercept_(1)+.1_dp)<1.e-14_dp, &
    'classifier warm average starts from published coefficients')
  c%warm_start=.false.;c%average=0;c%epsilon=.2_dp
  call c%fit(z,[1,2],coef_init=[0._dp],intercept_init=0._dp)
  call require(abs(c%coef_(1,1)-.2_dp)<1.e-14_dp,'binary vector coefficient and scalar intercept initializers')
  do i=1,4
    c%loss=losses(i);call c%fit(z,[1,2]);values=c%decision_function(z)
    call require(all(ieee_is_finite(values)),'regression-derived classifier losses')
  end do
  c%loss='modified_huber';c%max_iter=20;call c%fit(z,[1,2]);values=c%predict_log_proba(z)
  call require(all(values<=0),'hard-zero probabilities produce negative infinite logs safely')
  print *, 'SGD warm starts, initialization and full classifier losses: PASS'
end program
