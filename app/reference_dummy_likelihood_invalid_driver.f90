program reference_dummy_likelihood_invalid_driver
  use sklearn_common,only:dp
  use sklearn_stateless_preprocessing,only:add_dummy_feature
  use sklearn_covariance_functions,only:log_likelihood
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use,intrinsic::ieee_arithmetic
  implicit none
  type(csr_matrix)::a,b
  real(dp)::x(2,2),wide(2,3),score
  real(dp),allocatable::z(:,:)
  integer::mode
  character(16)::arg
  call get_command_argument(1,arg);read(arg,*)mode
  x=1;wide=1
  select case(mode)
  case(1)
    z=add_dummy_feature(x(:0,:))
  case(2)
    z=add_dummy_feature(x(:,:0))
  case(3)
    z=add_dummy_feature(x,ieee_value(0._dp,ieee_quiet_nan))
  case(4)
    z=add_dummy_feature(x,ieee_value(0._dp,ieee_positive_inf))
  case(5)
    x(1,1)=ieee_value(0._dp,ieee_quiet_nan);z=add_dummy_feature(x)
  case(6)
    x(1,1)=ieee_value(0._dp,ieee_positive_inf);z=add_dummy_feature(x)
  case(7)
    b=add_dummy_feature(a)
  case(8)
    a=csr_from_dense(x);a%data(1)=ieee_value(0._dp,ieee_quiet_nan);b=add_dummy_feature(a)
  case(9)
    a=csr_from_dense(x);a%data(1)=ieee_value(0._dp,ieee_positive_inf);b=add_dummy_feature(a)
  case(10)
    a=csr_from_dense(x);b=add_dummy_feature(a,ieee_value(0._dp,ieee_quiet_nan))
  case(11)
    a=csr_from_dense(x(:0,:));b=add_dummy_feature(a)
  case(12)
    a=csr_from_dense(x(:,:0));b=add_dummy_feature(a)
  case(13)
    a=csr_from_dense(x);a%indices(1)=0;b=add_dummy_feature(a)
  case(14)
    a=csr_from_dense(x);a%indptr(2)=0;b=add_dummy_feature(a)
  case(15)
    a=csr_from_arrays(1,huge(1),[1,1],[integer::],[real(dp)::]);b=add_dummy_feature(a)
  case(16)
    score=log_likelihood(wide,wide)
  case(17)
    score=log_likelihood(wide,x)
  case(18)
    a=csr_from_dense(x);b=add_dummy_feature(a,ieee_value(0._dp,ieee_negative_inf))
  case(19)
    score=log_likelihood(wide(:0,:),wide(:0,:))
  case(20)
    score=log_likelihood(x(:0,:0),x)
  end select
  print *,'guard unexpectedly returned'
end program
