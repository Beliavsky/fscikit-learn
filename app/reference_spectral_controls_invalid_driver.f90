program reference_spectral_controls_invalid_driver
  use sklearn_common,only:dp
  use sklearn_cluster,only:spectral_clustering
  use sklearn_spectral,only:spectral_embedding
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(spectral_clustering)::m
  type(spectral_embedding)::e
  type(csr_matrix)::s
  real(dp)::x(6,6)
  integer::which
  character(20)::arg
  call get_command_argument(1,arg);read(arg,*)which
  x=1;s=csr_from_dense(x,store_zeros=.true.)
  m%n_clusters=2;m%n_components=2;m%affinity='precomputed';m%assign_labels='cluster_qr';m%n_neighbors=3
  select case(which)
  case(1);m%n_init=0;call m%fit(x)
  case(2);m%n_init=0;m%assign_labels='discretize';call m%fit(s)
  case(3);m%gamma=-1;call m%fit(x)
  case(4);m%degree=-.5_dp;call m%fit(s)
  case(5);m%coef0=ieee_value(0._dp,ieee_quiet_nan);call m%fit(x)
  case(6);m%degree=ieee_value(0._dp,ieee_positive_inf);call m%fit(s)
  case(7);m%n_neighbors=0;call m%fit(x)
  case(8);m%random_state=-1;call m%fit(s)
  case(9);m%n_clusters=0;call m%fit(x)
  case(10);m%n_components=-1;call m%fit(s)
  case(11);m%n_components=6;call m%fit(s)
  case(12);m%n_components=8;call m%fit(s)
  case(13);x=-1;call m%fit(x)
  case(14);m%affinity='chi2';call m%fit(s)
  case(15);m%affinity='chi2';x(1,1)=-1;call m%fit(x)
  case(16);m%degree=ieee_value(0._dp,ieee_quiet_nan);m%affinity='linear';call m%fit(x)
  case(17);e%n_components=6;e%affinity='precomputed';call e%fit(s)
  case(18);e%affinity='linear';call e%fit(x)
  case(19);m%n_clusters=20;m%n_components=1;call m%fit(x)
  case(20);m%n_components=8;call m%fit(x)
  case(21);e%n_components=8;e%affinity='rbf';call e%fit(s)
  end select
  print *,'ACCEPTED'
end program
