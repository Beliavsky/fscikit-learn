program reference_minibatch_sparse_invalid_driver
  use sklearn_common,only:dp
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_minibatch_nmf,only:mini_batch_nmf
  use,intrinsic::ieee_arithmetic
  implicit none
  type(mini_batch_nmf)::model
  type(csr_matrix)::x,q
  real(dp)::a(4,3)
  real(dp),allocatable::w(:,:),w0(:,:),h0(:,:),dense_x(:,:),dense_q(:,:)
  integer::which
  logical::partial,dense
  character(32)::arg
  call get_command_argument(1,arg);read(arg,*)which
  call get_command_argument(2,arg);dense=trim(arg)=='dense'
  a=reshape([1._dp,.5_dp,0._dp,1._dp,.2_dp,0._dp,.7_dp,.4_dp,.8_dp,.9_dp,.3_dp,0._dp],[4,3])
  x=csr_from_dense(a);q=x;allocate(w0(4,2),h0(2,3));w0=.3_dp;h0=.4_dp;partial=.false.
  model%init='random';model%n_components=2;model%max_iter=3;model%batch_size=2
  select case(which)
  case(0);model%max_iter=0
  case(1);model%batch_size=0
  case(2);model%tol=-.1_dp
  case(3);model%tol=ieee_value(0._dp,ieee_quiet_nan)
  case(4);model%alpha_w=-.1_dp
  case(5);model%alpha_h=-.2_dp
  case(6);model%l1_ratio=2
  case(7);model%forget_factor=-.1_dp
  case(8);model%forget_factor=1.1_dp
  case(9);model%fresh_restarts_max_iter=0
  case(10);model%transform_max_iter=-1
  case(11);model%max_no_improvement=0
  case(12);model%max_no_improvement=-2
  case(13);model%random_state=-1
  case(14);model%n_components=-1
  case(15);model%init='bad'
  case(16);model%beta_loss=ieee_value(0._dp,ieee_quiet_nan)
  case(17);x%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(18);x%data(1)=ieee_value(0._dp,ieee_positive_inf)
  case(19);x%data(1)=-.5_dp
  case(20);x=csr_from_arrays(0,3,[1],[integer::],[real(dp)::])
  case(21);x=csr_from_arrays(4,0,[1,1,1,1,1],[integer::],[real(dp)::])
  case(22);q%n_cols=4
  case(23);q%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(24);q=csr_from_arrays(0,3,[1],[integer::],[real(dp)::])
  case(25)
    print '(a)','minibatch_guard query'
    if(dense)then
      dense_q=q%to_dense();w=model%transform(dense_q)
    else
      w=model%transform(q)
    end if
    stop
  case(26);model%init='custom';h0=0
  case(27);model%init='custom';w0(1,1)=-1
  case(28);model%init='custom';h0=reshape([.3_dp,.4_dp,.5_dp,.6_dp],[2,2])
  case(29,37,38)
    x=csr_from_arrays(4,3,[1,3,5,7,9],[1,1,2,2,3,3,1,1], &
      [1._dp,-.5_dp,1._dp,-.2_dp,1._dp,-.3_dp,2._dp,-.5_dp])
    partial=which/=29
    if(which==38)model%init='custom'
  case(30);model%beta_loss=0
  case(31);model%beta_loss=1;x=csr_from_dense(a,.true.)
  case(32)
    model%beta_loss=1;q=csr_from_arrays(1,3,[1,2],[1],[0._dp])
  case(33);model%beta_loss=0;partial=.true.
  case(34);model%init='custom';x%data(1)=-.1_dp;partial=.true.
  case(35);model%beta_loss=0;x=csr_from_dense(a+1)
  case(39,40);q%data=-q%data
  case(41);model%init='nndsvd';model%n_components=5
  case(42)
    model%beta_loss=1;partial=.true.;x=csr_from_arrays(1,3,[1,2],[1],[0._dp]);w0=reshape([.3_dp,.3_dp],[1,2])
  end select
  if(dense)then
    dense_x=x%to_dense();dense_q=q%to_dense()
  end if
  if(partial)then
    print '(a)','minibatch_guard partial'
    if(dense)then
      call model%partial_fit(dense_x,w0,h0)
    else
      call model%partial_fit(x,w0,h0)
    end if
  else
    print '(a)','minibatch_guard fit'
    if(dense)then
      w=model%fit_transform(dense_x,w0,h0)
    else
      w=model%fit_transform(x,w0,h0)
    end if
  end if
  if(which==36.or.which==39)then
    if(which==36)then
      model%beta_loss=.5_dp;model%n_components=1;model%batch_size=1;model%forget_factor=0;model%transform_max_iter=1
    end if
    print '(a)','minibatch_guard partial'
    if(dense)then
      call model%partial_fit(dense_q)
    else
      call model%partial_fit(q)
    end if
  end if
  print '(a)','minibatch_guard query'
  if(dense)then
    w=model%transform(dense_q)
  else
    w=model%transform(q)
  end if
  print '(a)','accepted'
end program
