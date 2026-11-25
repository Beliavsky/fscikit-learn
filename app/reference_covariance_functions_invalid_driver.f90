program reference_covariance_functions_invalid_driver
  use sklearn_common,only:dp
  use sklearn_covariance,only:empirical_covariance,shrunk_covariance,ledoit_wolf,oas,covariance_estimate
  use,intrinsic::ieee_arithmetic
  implicit none
  real(dp)::x(3,2),a(2,2),nan
  real(dp),allocatable::cov(:,:),batch(:,:,:)
  type(covariance_estimate)::estimate
  integer::which
  character(16)::arg
  call get_command_argument(1,arg);read(arg,*)which
  x=1;a=1;nan=ieee_value(0._dp,ieee_quiet_nan)
  select case(which)
  case(1);cov=empirical_covariance(x(:0,:))
  case(2);cov=empirical_covariance(x(:,:0))
  case(3);cov=empirical_covariance([real(dp)::])
  case(4);cov=shrunk_covariance(a,-1._dp)
  case(5);cov=shrunk_covariance(a,2._dp)
  case(6);cov=shrunk_covariance(a,nan)
  case(7);a(1,1)=nan;cov=shrunk_covariance(a)
  case(8);cov=shrunk_covariance(x)
  case(9);cov=shrunk_covariance(a(:0,:))
  case(10);estimate=ledoit_wolf(x,block_size=0)
  case(11);x(1,1)=nan;estimate=ledoit_wolf(x)
  case(12);x(1,1)=ieee_value(0._dp,ieee_positive_inf);estimate=oas(x)
  case(13);estimate=ledoit_wolf(x(:0,:))
  case(14);estimate=oas(x(:,:0))
  case(15);x=huge(1._dp)/2;estimate=oas(x,assume_centered=.true.)
  case(16);x=huge(1._dp)/2;estimate=ledoit_wolf(x,assume_centered=.true.)
  case default;error stop 'unknown covariance function guard'
  end select
  print *,'covariance function guard unexpectedly returned'
end program
