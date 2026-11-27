program reference_ard_invalid_driver
  use sklearn_linear_model,only:ard_regression
  use sklearn_common,only:dp
  use,intrinsic::ieee_arithmetic
  implicit none
  type(ard_regression)::m
  real(dp)::x(4,2),y(4),weights(4),std(4)
  integer::which
  character(20)::arg
  call get_command_argument(1,arg);read(arg,*)which
  x=reshape([.1_dp,.2_dp,.3_dp,.4_dp,.4_dp,.1_dp,.3_dp,.2_dp],[4,2]);y=[1,2,3,4];weights=1
  select case(which)
  case(0);m%tol=-1
  case(1);m%tol=ieee_value(0._dp,ieee_quiet_nan)
  case(2);m%max_iter=0
  case(3);m%alpha_1=-1
  case(4);m%alpha_2=-1
  case(5);m%lambda_1=-1
  case(6);m%lambda_2=-1
  case(7);m%threshold_lambda=-1
  case(8);m%threshold_lambda=ieee_value(0._dp,ieee_positive_inf)
  case(9);m%lambda_2=ieee_value(0._dp,ieee_quiet_nan)
  case(10);call m%fit(x(:1,:),y(:1));stop 0
  case(11);call m%fit(x,spread(y,2,2));stop 0
  case(12);call m%fit(x,y,weights);stop 0
  case(13);x(1,1)=ieee_value(0._dp,ieee_positive_inf)
  case(14);y(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(15);std=m%predict_std(x);print *,std;stop 0
  end select
  call m%fit(x,y)
end program
