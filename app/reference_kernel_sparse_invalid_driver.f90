program reference_kernel_sparse_invalid_driver
  use sklearn_kernel_pca,only:kernel_pca
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use sklearn_common,only:dp
  use,intrinsic::ieee_arithmetic
  implicit none
  type(kernel_pca)::model
  type(csr_matrix)::s,q
  real(dp)::x(4,2)
  real(dp),allocatable::z(:,:)
  integer::which
  character(20)::arg
  call get_command_argument(1,arg);read(arg,*)which
  x=reshape([.1_dp,.2_dp,.3_dp,.4_dp,.4_dp,.1_dp,.3_dp,.2_dp],[4,2]);s=csr_from_dense(x);q=s
  model%n_components=2
  select case(which)
  case(0);model%fit_inverse_transform=.true.
  case(1);model%kernel='precomputed';s=csr_from_dense(matmul(x,transpose(x)))
  case(2)
    model%kernel='precomputed';call model%fit(matmul(x,transpose(x)));q=csr_from_dense(matmul(x,transpose(x)))
    z=model%transform(q);print *,z;stop 0
  case(3);s%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(4);s%data(1)=ieee_value(0._dp,ieee_positive_inf)
  case(5);call model%fit(s);q=csr_from_dense(x(:,:1));z=model%transform(q);print *,z;stop 0
  case(6);call model%fit(s);q%data(1)=ieee_value(0._dp,ieee_quiet_nan);z=model%transform(q);print *,z;stop 0
  case(7);z=model%transform(s);print *,z;stop 0
  case(8);s=csr_from_dense(reshape([real(dp)::],[0,2]))
  case(9);s=csr_from_dense(reshape([real(dp)::],[4,0]))
  case(10);call model%fit(s);q=csr_from_dense(reshape([real(dp)::],[0,2]));z=model%transform(q);print *,z;stop 0
  case(11);call model%fit(s);q%data(1)=ieee_value(0._dp,ieee_positive_inf);z=model%transform(q);print *,z;stop 0
  end select
  call model%fit(s)
end program
