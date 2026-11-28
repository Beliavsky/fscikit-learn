program reference_nmf_sparse_invalid_driver
  use sklearn_common,only:dp
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_decomposition,only:nmf
  use,intrinsic::ieee_arithmetic
  implicit none
  type(nmf)::model
  type(csr_matrix)::x,q
  real(dp)::a(4,3)
  real(dp),allocatable::w(:,:),w0(:,:),h0(:,:)
  integer::which
  character(32)::arg
  call get_command_argument(1,arg);read(arg,*)which
  a=reshape([1._dp,.5_dp,0._dp,1._dp,.2_dp,0._dp,.7_dp,.4_dp,.8_dp,.9_dp,.3_dp,0._dp],[4,3])
  x=csr_from_dense(a);q=x;allocate(w0(4,2),h0(2,3));w0=.3_dp;h0=.4_dp
  model%solver='cd';model%init='random';model%n_components=2;model%max_iter=5
  select case(which)
  case(0);model%max_iter=0
  case(1);model%tol=ieee_value(0._dp,ieee_quiet_nan)
  case(2);model%beta_loss=ieee_value(0._dp,ieee_positive_inf)
  case(3);model%solver='bad'
  case(4);model%init='bad'
  case(5);model%random_state=-1
  case(6);model%n_components=-1
  case(7);model%alpha_w=-.1_dp
  case(8);model%alpha_h=-.2_dp
  case(9);model%l1_ratio=2
  case(10);x%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(11);x%data(1)=ieee_value(0._dp,ieee_positive_inf)
  case(12);x%data(1)=-.5_dp
  case(13);x=csr_from_arrays(0,3,[1],[integer::],[real(dp)::])
  case(14);x=csr_from_arrays(4,0,[1,1,1,1,1],[integer::],[real(dp)::])
  case(15);model%init='custom';h0=0
  case(16);model%init='custom';h0(1,1)=-1
  case(17);model%init='custom';h0=reshape([.3_dp,.4_dp,.5_dp,.6_dp],[2,2])
  case(18);q%n_cols=4
  case(19);q%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(20);q=csr_from_arrays(0,3,[1],[integer::],[real(dp)::])
  case(21)
    print '(a)','nmf_stage query';w=model%transform(q);stop
  case(22);model%solver='mu';model%beta_loss=0
  case(23);model%solver='mu';model%beta_loss=1;x=csr_from_dense(a,.true.)
  case(24);model%n_components=5;model%init='nndsvd'
  case(25);model%beta_loss=1
  case(26);model%init='custom';w0=0
  case(27)
    x=csr_from_arrays(4,3,[1,3,5,7,9],[1,1,2,2,3,3,1,1], &
      [1._dp,-.5_dp,1._dp,-.2_dp,1._dp,-.3_dp,2._dp,-.5_dp])
  end select
  print '(a)','nmf_stage fit';w=model%fit_transform(x,w0,h0)
  print '(a)','nmf_stage query';w=model%transform(q)
  print '(a)','accepted'
end program
