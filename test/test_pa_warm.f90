program test_pa_warm
  use sklearn_common,only:dp,require
  use sklearn_linear_model,only:passive_aggressive_regressor,passive_aggressive_classifier
  use sklearn_sgd_clone,only:clone_pa_parameters
  use, intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(passive_aggressive_regressor)::r,cold
  type(passive_aggressive_classifier)::c
  real(dp)::x(4,1),y(4),bad
  integer::labels(4)
  character(40)::mode
  x(:,1)=[1._dp,2._dp,-1._dp,-2._dp];y=[1._dp,0._dp,-1._dp,0._dp];labels=[0,1,0,1]
  r%c=.01_dp;r%max_iter=1;r%tol=-1;r%shuffle=.false.;r%warm_start=.true.;r%fit_intercept=.false.
  c%max_iter=1;c%tol=-1;c%shuffle=.false.;c%warm_start=.true.
  call get_command_argument(1,mode)
  if(index(mode,'average')>0)r%average=1
  call r%fit(x,y);call c%fit(x,labels)
  bad=ieee_value(0._dp,ieee_quiet_nan)
  if(index(mode,'inf')>0)bad=ieee_value(0._dp,ieee_positive_inf)
  select case(trim(mode))
  case('reg_width');call r%fit(reshape([x,x],[4,2]),y)
  case('class_width');call c%fit(reshape([x,x],[4,2]),labels)
  case('class_heads');call c%fit(x,[0,1,2,0])
  case('reg_nan','reg_inf');r%coef_=bad;call r%fit(x,y)
  case('reg_bias_nan');r%intercept_=bad;call r%fit(x,y)
  case('class_nan','class_inf');c%coef_=bad;call c%fit(x,labels)
  case('class_bias_nan');c%intercept_=bad;call c%fit(x,labels)
  case('class_bias_shape');c%intercept_=[0._dp,0._dp];call c%fit(x,labels)
  case('class_bias_absent');deallocate(c%intercept_);call c%fit(x,labels)
  case('reg_enable_average');r%average=1;call r%fit(x,y)
  case('reg_average_nan');r%coef_=bad;r%intercept_=bad;call r%fit(x,y)
    call require(all(abs(r%coef_)<huge(1._dp)),'averaged regressor ignores replaced public parameters')
    print *,'Accepted averaged public replacement: PASS';stop
  end select
  call require(len_trim(mode)==0,'PA warm guard unexpectedly accepted')
  cold=clone_pa_parameters(r)
  call require(.not.allocated(cold%coef_).and.cold%warm_start,'clone drops fitted state and retains warm control')
  call cold%fit(x,y);call r%fit(x,y)
  call require(r%t_==5.and.cold%t_==5,'warm fit resets sample clock')
  call require(any(r%coef_/=cold%coef_),'warm fit continues coefficients')
  r%warm_start=.false.;call r%fit(x,y)
  call require(all(r%coef_==cold%coef_),'cold fit discards previous state')
  c%average=1;call c%fit(x,labels)
  call require(c%t_==5,'warm classifier can enable averaging')
  c%warm_start=.false.;c%coef_=bad;call c%fit(x,labels)
  call require(all(abs(c%coef_)<huge(1._dp)),'cold classifier ignores previous public parameters')
  print *,'PA warm/cold state, clocks, averaging and parameter clones: PASS'
end program
