program reference_shrinkage_invalid_driver
  use sklearn_common,only:dp
  use sklearn_covariance,only:ledoit_wolf_shrinkage,ledoit_wolf
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  real(dp)::x(2,2),value
  integer::which
  character(20)::argument
  type(ledoit_wolf)::model
  call get_command_argument(1,argument);read(argument,*)which
  x=1
  select case(which)
  case(1)
    value=ledoit_wolf_shrinkage(x,block_size=0)
  case(2)
    value=ledoit_wolf_shrinkage(x,block_size=-1)
  case(3)
    x=ieee_value(1._dp,ieee_quiet_nan);value=ledoit_wolf_shrinkage(x)
  case(4)
    value=ledoit_wolf_shrinkage(x(:0,:))
  case(5)
    value=ledoit_wolf_shrinkage(x(:,:0))
  case(6)
    model%block_size=0;call model%fit(x)
  case(7)
    value=ledoit_wolf_shrinkage(x(:,:1),block_size=0)
  end select
end program
