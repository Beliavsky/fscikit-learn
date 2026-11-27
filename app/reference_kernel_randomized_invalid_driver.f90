program reference_kernel_randomized_invalid_driver
  use sklearn_kernel_pca,only:kernel_pca
  use sklearn_common,only:dp
  use,intrinsic::ieee_arithmetic
  implicit none
  type(kernel_pca)::m
  real(dp)::x(4,2)
  real(dp),allocatable::z(:,:)
  integer::which
  character(20)::arg
  call get_command_argument(1,arg);read(arg,*)which
  x=reshape([.1_dp,.2_dp,.3_dp,.4_dp,.4_dp,.1_dp,.3_dp,.2_dp],[4,2]);m%eigen_solver='randomized';m%n_components=2
  select case(which)
  case(0);m%eigen_solver='unknown'
  case(1);m%iterated_power=-2
  case(2);m%random_state=-1
  case(3);m%n_components=-1
  case(4);m%gamma=ieee_value(0._dp,ieee_quiet_nan)
  case(5);z=m%transform(x);print *,z;stop 0
  case(6);x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
  case(7);x(1,1)=ieee_value(0._dp,ieee_positive_inf)
  case(8);m%degree=-1
  case(9);m%fit_inverse_transform=.true.;m%kernel='precomputed';call m%fit(matmul(x,transpose(x)));stop 0
  case(10);call m%fit(x);z=m%transform(x(:,:1));print *,z;stop 0
  case(11);call m%fit(x);x(1,1)=ieee_value(0._dp,ieee_quiet_nan);z=m%transform(x);print *,z;stop 0
  end select
  call m%fit(x)
end program
