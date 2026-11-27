program reference_quantile_csr_invalid_driver
  use sklearn_quantile,only:quantile_regressor
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use sklearn_common,only:dp
  use,intrinsic::ieee_arithmetic
  implicit none
  type(quantile_regressor)::m
  type(csr_matrix)::s
  real(dp)::x(4,2),y(4),w(4),pred(4)
  integer::which
  character(20)::arg
  call get_command_argument(1,arg);read(arg,*)which
  x=reshape([.1_dp,.2_dp,.3_dp,.4_dp,.4_dp,.1_dp,.3_dp,.2_dp],[4,2]);y=[1,2,3,4];w=1;s=csr_from_dense(x)
  select case(which)
  case(0);m%quantile=0
  case(1);m%quantile=1
  case(2);m%quantile=ieee_value(0._dp,ieee_quiet_nan)
  case(3);m%alpha=-1
  case(4);m%alpha=ieee_value(0._dp,ieee_positive_inf)
  case(5);call m%fit(s,y,w(:3));stop 0
  case(6);w(1)=-1
  case(7);w(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(8);w(1)=ieee_value(0._dp,ieee_positive_inf)
  case(9);call m%fit(s,spread(y,2,2));stop 0
  case(10);s%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(11);s%data(1)=ieee_value(0._dp,ieee_positive_inf)
  case(12);pred=m%predict(s);print *,pred;stop 0
  case(13);call m%fit(s,y);s=csr_from_dense(x(:,:1));pred=m%predict(s);print *,pred;stop 0
  case(14);s=csr_from_dense(reshape([real(dp)::],[0,2]));call m%fit(s,[real(dp)::]);stop 0
  case(15);s=csr_from_dense(reshape([real(dp)::],[4,0]))
  end select
  call m%fit(s,y,w)
end program
