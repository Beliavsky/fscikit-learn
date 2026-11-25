program reference_anova_invalid_driver
  use sklearn_common,only:dp
  use sklearn_univariate_selection,only:f_oneway,anova_sample,anova_result
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic
  implicit none
  type(anova_sample)::samples(2),empty(0),one(1)
  type(anova_result)::test
  type(csr_matrix)::a
  real(dp)::x(2,2),y(2,3)
  character(16)::arg
  integer::mode
  call get_command_argument(1,arg);read(arg,*)mode
  x=1;y=2
  select case(mode)
  case(1)
    test=f_oneway(empty)
  case(2)
    test=f_oneway(one)
  case(3)
    samples(1)=anova_sample(x(:0,:))
  case(4)
    samples(1)=anova_sample(x(:,:0))
  case(5)
    test=f_oneway(x,y)
  case(6)
    x(1,1)=ieee_value(0._dp,ieee_quiet_nan);samples(1)=anova_sample(x)
  case(7)
    x(1,1)=ieee_value(0._dp,ieee_positive_inf);samples(1)=anova_sample(x)
  case(8)
    samples(1)=anova_sample(a)
  case(9)
    a=csr_from_dense(x);a%indices(1)=0;samples(1)=anova_sample(a)
  case(10)
    a=csr_from_dense(x);a%data(1)=ieee_value(0._dp,ieee_quiet_nan);samples(1)=anova_sample(a)
  case(11)
    a=csr_from_dense(x);a%data(1)=ieee_value(0._dp,ieee_negative_inf);samples(1)=anova_sample(a)
  case(12)
    a=csr_from_dense(x);samples=[anova_sample(x),anova_sample(a)];test=f_oneway(samples)
  end select
  print *,'guard unexpectedly returned'
end program
