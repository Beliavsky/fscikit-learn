program reference_structured_ward_invalid_driver
  use sklearn_common,only:dp
  use sklearn_cluster,only:agglomerative_clustering,feature_agglomeration
  use sklearn_sparse,only:csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(agglomerative_clustering)::model
  type(feature_agglomeration)::features
  real(dp)::x(3,2),g(3,3)
  integer::which
  character(20)::argument
  call get_command_argument(1,argument);read(argument,*)which
  x=1;g=1
  select case(which)
  case(1);model%n_clusters=0
  case(2);model%distance_threshold=1
  case(3);model%n_clusters=0;model%distance_threshold=-1
  case(4);model%n_clusters=0;model%distance_threshold=ieee_value(0._dp,ieee_quiet_nan)
  case(5);model%n_clusters=0;model%distance_threshold=ieee_value(0._dp,ieee_positive_inf)
  case(6);model%n_clusters=0;model%distance_threshold=1;model%compute_full_tree='false'
  case(7);model%compute_full_tree='bad'
  case(8);model%connectivity=g;model%connectivity_mask=g/=0
  case(9);model%connectivity=g;model%connectivity_csr=csr_from_dense(g)
  case(10);model%connectivity=g(:2,:2)
  case(11);model%connectivity_mask=g(:2,:2)/=0
  case(12);model%connectivity_csr=csr_from_dense(g(:2,:2))
  case(13);model%n_clusters=4
  case(14);model%n_clusters=-1
  case(15);call model%fit(x(:1,:));stop
  case(16);call model%fit(x(:0,:));stop
  case(17);x=ieee_value(0._dp,ieee_quiet_nan)
  case(18);model%linkage='single';model%connectivity=g
  case(19);model%linkage='single';model%n_clusters=0;model%distance_threshold=1
  case(20);features%n_clusters=0;call features%fit(x);stop
  case(21);features%connectivity=g;call features%fit(x);stop
  case(22);model%connectivity=ieee_value(g,ieee_quiet_nan)
  case(23);x(1,:)=1.e200_dp;x(2,:)=-1.e200_dp
  end select
  call model%fit(x)
end program
