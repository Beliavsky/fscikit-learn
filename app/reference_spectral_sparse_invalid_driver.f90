program reference_spectral_sparse_invalid_driver
  use sklearn_common,only:dp
  use sklearn_spectral,only:spectral_embedding
  use sklearn_cluster,only:spectral_clustering
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(spectral_embedding)::m
  type(spectral_clustering)::c
  type(csr_matrix)::s
  real(dp)::x(6,6)
  integer::i,j,which
  character(20)::arg
  call get_command_argument(1,arg);read(arg,*)which
  do i=1,6
    do j=1,6
      x(i,j)=abs(real(i-j,dp))
    end do
  end do
  m%n_components=2;m%n_neighbors=3;m%affinity='precomputed_nearest_neighbors'
  s=csr_from_dense(x,store_zeros=.true.)
  select case(which)
  case(1);m%n_neighbors=0;call m%fit(s)
  case(2);m%n_neighbors=7;call m%fit(s)
  case(3);s=csr_from_dense(x(:,:5));call m%fit(s)
  case(4);s%data(1)=-1;call m%fit(s)
  case(5);s%data(1)=ieee_value(0._dp,ieee_quiet_nan);call m%fit(s)
  case(6);s%data(1)=ieee_value(0._dp,ieee_positive_inf);call m%fit(s)
  case(7);s=csr_from_dense(x(:0,:));call m%fit(s)
  case(8);s=csr_from_dense(x(:1,:));m%affinity='rbf';call m%fit(s)
  case(9);m%affinity='precomputed';s=csr_from_dense(x(:,:5));call m%fit(s)
  case(10);m%affinity='rbf';m%gamma=-2;call m%fit(s)
  case(11)
    s=csr_from_dense(x(:,:0));m%affinity='rbf';call m%fit(s)
  case(12)
    s=csr_matrix();s%n_rows=6;s%n_cols=6;s%indptr=[1,2,3,4,5,6,7]
    s%indices=[1,2,3,4,5,6];s%data=[(0._dp,i=1,6)]
    call m%fit(s)
  case(13);c%n_clusters=2;c%n_neighbors=7;c%affinity='precomputed_nearest_neighbors';call c%fit(s)
  case(14);c%n_clusters=2;c%affinity='precomputed';s=csr_from_dense(x(:,:5));call c%fit(s)
  end select
  print *,'UNEXPECTED ACCEPTANCE'
end program
