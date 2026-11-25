program reference_ward_invalid_driver
  use sklearn_common,only:dp
  use sklearn_hierarchy,only:ward_tree,hierarchy_result
  use sklearn_sparse,only:csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  real(dp)::x(3,2),g(3,3)
  integer::which
  character(20)::argument
  type(hierarchy_result)::tree
  call get_command_argument(1,argument);read(argument,*)which
  x=1;g=1
  select case(which)
  case(1)
    tree=ward_tree(x(:0,:))
  case(2)
    tree=ward_tree(x(:,:0))
  case(3)
    x=ieee_value(1._dp,ieee_quiet_nan);tree=ward_tree(x)
  case(4)
    tree=ward_tree(x(:1,:))
  case(5)
    tree=ward_tree(x,g(:2,:2))
  case(6)
    tree=ward_tree(x,g(:2,:2)/=0)
  case(7)
    tree=ward_tree(x,csr_from_dense(g(:2,:2)))
  case(8)
    g=ieee_value(1._dp,ieee_quiet_nan);tree=ward_tree(x,g)
  case(9)
    g=ieee_value(1._dp,ieee_quiet_nan);tree=ward_tree(x,csr_from_dense(g))
  case(10)
    tree=ward_tree(x,g,n_clusters=-1)
  case(11)
    tree=ward_tree(x,g,n_clusters=4)
  case(12)
    tree=ward_tree(x,g/=0,n_clusters=4)
  case(13)
    tree=ward_tree(x,csr_from_dense(g),n_clusters=-1)
  case(14)
    x(1,:)=1.e200_dp;x(2,:)=-1.e200_dp;tree=ward_tree(x)
  case(15)
    x(1,:)=1.e200_dp;x(2,:)=-1.e200_dp;tree=ward_tree(x,g)
  end select
end program
