program reference_linkage_invalid_driver
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
  case(1);tree=linkage_tree(x(:0,:))
  case(2);tree=linkage_tree(x(:,:0))
  case(3);x=ieee_value(x,ieee_quiet_nan);tree=linkage_tree(x)
  case(4);tree=linkage_tree(x(:1,:))
  case(5);tree=linkage_tree(x,linkage='unknown')
  case(6);tree=linkage_tree(x,affinity='unknown')
  case(7);x(1,:)=0;tree=linkage_tree(x,affinity='cosine')
  case(8);tree=linkage_tree(x,affinity='precomputed')
  case(9);tree=linkage_tree(x,g(:2,:2))
  case(10);tree=linkage_tree(x,g(:2,:2)/=0)
  case(11);tree=linkage_tree(x,csr_from_dense(g(:2,:2)))
  case(12);g=ieee_value(g,ieee_quiet_nan);tree=linkage_tree(x,g)
  case(13);g=ieee_value(g,ieee_quiet_nan);tree=linkage_tree(x,csr_from_dense(g))
  case(14);tree=linkage_tree(x,g,n_clusters=-1)
  case(15);tree=linkage_tree(x,g,n_clusters=3)
  case(16);tree=linkage_tree(x,g,n_clusters=4)
  case(17);tree=linkage_tree(x,g,affinity='chebyshev')
  case(18);tree=linkage_tree(x,g,affinity='sqeuclidean')
  case(19);tree=linkage_tree(x,g,affinity='minkowski')
  case(20);x(1,:)=1.e200_dp;x(2,:)=-1.e200_dp;tree=linkage_tree(x)
  case(21);model%linkage='complete';model%connectivity=g;model%n_clusters=3;model%compute_full_tree='false'
    call model%fit(x)
  case(22);model%metric='l2';call model%fit(x)
  case(23);model%linkage='single';model%metric='unknown';call model%fit(x)
  case(24);tree=linkage_tree(x,affinity='euclidean        invalid')
  case(25);tree=linkage_tree(x,linkage='complete         invalid')
  end select
end program
