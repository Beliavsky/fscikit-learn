program reference_single_invalid_driver
  use sklearn_common,only:dp
  use sklearn_hierarchy,only:linkage_tree,hierarchy_result
  use sklearn_cluster,only:agglomerative_clustering
  use sklearn_sparse,only:csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(hierarchy_result)::tree
  type(agglomerative_clustering)::model
  real(dp)::x(3,2),g(3,3)
  integer::which
  character(20)::argument
  call get_command_argument(1,argument);read(argument,*)which
  x=1;g=1
  select case(which)
  case(1);tree=linkage_tree(x(:0,:),linkage='single')
  case(2);tree=linkage_tree(x(:,:0),linkage='single')
  case(3);x=ieee_value(x,ieee_quiet_nan);tree=linkage_tree(x,linkage='single')
  case(4);tree=linkage_tree(x(:1,:1),linkage='single',affinity='precomputed')
  case(5);tree=linkage_tree(x(:1,:),g(:1,:1),linkage='single')
  case(6);tree=linkage_tree(x(:1,:),g(:1,:1),linkage='single',affinity='cosine')
  case(7);model%linkage='single';model%metric='precomputed';call model%fit(x(:1,:1))
  case(8);tree=linkage_tree(x,g(:2,:2),linkage='single')
  case(9);tree=linkage_tree(x,g(:2,:2)/=0,linkage='single')
  case(10);tree=linkage_tree(x,csr_from_dense(g(:2,:2)),linkage='single')
  case(11);g=ieee_value(g,ieee_quiet_nan);tree=linkage_tree(x,g,linkage='single')
  case(12);g=ieee_value(g,ieee_quiet_nan);tree=linkage_tree(x,csr_from_dense(g),linkage='single')
  case(13);tree=linkage_tree(x,g,n_clusters=-1,linkage='single')
  case(14);tree=linkage_tree(x,g,n_clusters=4,linkage='single')
  case(15);tree=linkage_tree(x,g,linkage='single',affinity='infinity')
  case(16);tree=linkage_tree(x,g,linkage='single',affinity='minkowski')
  case(17);model%linkage='single';model%metric='p';call model%fit(x)
  case(18);model%linkage='single';model%metric='infinity';call model%fit(x)
  case(19);x(1,:)=0;tree=linkage_tree(x,linkage='single',affinity='cosine')
  case(20);x(1,:)=1.e200_dp;x(2,:)=-1.e200_dp;tree=linkage_tree(x,linkage='single',affinity='sqeuclidean')
  end select
end program
