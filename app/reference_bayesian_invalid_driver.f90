program reference_bayesian_invalid_driver
  use sklearn_linear_model,only:bayesian_ridge
  use sklearn_common,only:dp
  use,intrinsic::ieee_arithmetic
  implicit none
  type(bayesian_ridge)::model
  real(dp)::x(4,2),y(4),w(4),value(4)
  integer::which
  character(32)::arg
  x(:,1)=[.1_dp,.2_dp,.3_dp,.4_dp];x(:,2)=[.4_dp,.1_dp,.3_dp,.2_dp]
  y=[1._dp,2._dp,3._dp,4._dp];w=1
  call get_command_argument(1,arg);read(arg,*)which
  select case(which)
  case(0);model%tol=0
  case(1);model%tol=ieee_value(0._dp,ieee_quiet_nan)
  case(2);model%max_iter=0
  case(3);model%alpha_1=-1
  case(4);model%alpha_2=-1
  case(5);model%lambda_1=-1
  case(6);model%lambda_2=-1
  case(7);model%alpha_init=-2
  case(8);model%lambda_init=-2
  case(9);model%alpha_init=ieee_value(0._dp,ieee_positive_inf)
  case(10);model%lambda_2=ieee_value(0._dp,ieee_quiet_nan)
  case(11);w(1)=-1
  case(12);w(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(13);w(1)=ieee_value(0._dp,ieee_positive_inf)
  case(14);call model%fit(x,y,w(:3));stop
  case(15);w=0
  case(16);call model%fit(x,reshape([y,y],[4,2]));stop
  case(17);y(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(18);x(1,1)=ieee_value(0._dp,ieee_positive_inf)
  case(19);value=model%predict_std(x);stop
  end select
  call model%fit(x,y,w)
  print *,'unexpected acceptance'
end program
