program reference_incremental_sparse_invalid_driver
  use sklearn_common,only:dp
  use sklearn_incremental_pca,only:incremental_pca
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(incremental_pca)::model
  type(csr_matrix)::s
  real(dp)::x(6,3)
  real(dp),allocatable::z(:,:)
  integer::which,i
  character(20)::arg
  call get_command_argument(1,arg);read(arg,*)which
  do i=1,6
    x(i,:)=[sin(real(i,dp)),cos(.3_dp*i),real(i,dp)]
  end do
  model%n_components=2;model%batch_size=3;s=csr_from_dense(x)
  select case(which)
  case(1);call model%partial_fit(s)
  case(2);s=csr_from_dense(x(:0,:));call model%fit(s)
  case(3);s=csr_from_dense(x(:,:0));call model%fit(s)
  case(4);s%data(1)=ieee_value(0._dp,ieee_quiet_nan);call model%fit(s)
  case(5);s%data(1)=ieee_value(0._dp,ieee_positive_inf);call model%fit(s)
  case(6);model%batch_size=-1;call model%fit(s)
  case(7);model%n_components=4;call model%fit(s)
  case(8);model%batch_size=1;call model%fit(s)
  case(9);z=model%transform(s)
  case(10);call model%fit(x);s=csr_from_dense(x(:,:2));z=model%transform(s)
  case(11);call model%fit(x);s%data(1)=ieee_value(0._dp,ieee_positive_inf);z=model%transform(s)
  case(12);call model%partial_fit(x);z=model%transform(s)
  end select
  print *,'UNEXPECTED ACCEPTANCE'
end program
