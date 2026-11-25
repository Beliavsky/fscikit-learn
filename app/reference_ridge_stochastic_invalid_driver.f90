program reference_ridge_stochastic_invalid_driver
  use sklearn_common,only:dp
  use sklearn_ridge_solver
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(ridge_solver_options)::options
  type(ridge_solution)::fit
  real(dp)::x(2,2),y(2),a,w(2)
  character(20)::argument
  integer::which
  call get_command_argument(1,argument);read(argument,*)which
  options%solver='sag';x=1;y=1;a=1;w=1
  select case(which)
  case(1)
    options%random_state=-1
  case(2)
    options%solver='saga';options%return_intercept=.true.
  case(3)
    options%solver='lsqr';options%return_intercept=.true.
  case(4)
    options%positive=.true.
  case(5)
    x=0
  case(6)
    x=0;a=0
  case(7)
    x=0;a=0;options%solver='saga'
  case(8)
    w=ieee_value(1._dp,ieee_quiet_nan)
  case(9)
    fit=ridge_regression(x,y,a,options,[1._dp])
    stop 0
  case(10)
    options%tol=ieee_value(1._dp,ieee_quiet_nan)
  case(11)
    a=-1
  case(12)
    x=ieee_value(1._dp,ieee_quiet_nan)
  case(13)
    y=ieee_value(1._dp,ieee_quiet_nan)
  case(14)
    fit=ridge_regression(x(:0,:),y(:0),a,options)
    stop 0
  case(15)
    options%max_iter=-2
  end select
  fit=ridge_regression(x,y,a,options,w)
end program
