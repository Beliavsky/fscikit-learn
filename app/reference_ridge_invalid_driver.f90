program reference_ridge_invalid_driver
  use sklearn_common,only:dp
  use sklearn_ridge_solver
  use sklearn_sparse,only:csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  real(dp)::x(2,2),y(2),alpha(1),weights(2)
  type(ridge_solver_options)::options
  type(ridge_solution)::fit
  character(16)::argument
  integer::which
  call get_command_argument(1,argument);read(argument,*)which
  x=1;y=1;alpha=1;weights=1
  select case(which)
  case(1)
    alpha=-1
  case(2)
    alpha=ieee_value(1._dp,ieee_quiet_nan)
  case(3)
    weights=-1
  case(4)
    weights=ieee_value(1._dp,ieee_quiet_nan)
  case(5)
    options%solver='unknown'
  case(6)
    options%positive=.true.;options%solver='lsqr'
  case(7)
    options%solver='lbfgs'
  case(8)
    options%tol=-1
  case(9)
    options%max_iter=-2
  case(10)
    options%solver='svd'
    fit=ridge_regression(csr_from_dense(x),y,alpha,options)
    stop 0
  case(11)
    fit=ridge_regression(x,y,[1._dp,2._dp],options)
    stop 0
  case(12)
    fit=ridge_regression(x,y,alpha,options,[1._dp])
    stop 0
  case(13)
    y=ieee_value(1._dp,ieee_quiet_nan)
  end select
  fit=ridge_regression(x,y,alpha,options,weights)
end program
