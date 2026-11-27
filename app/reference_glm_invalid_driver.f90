program reference_glm_invalid_driver
  use sklearn
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(tweedie_regressor)::model
  type(csr_matrix)::sparse
  real(dp)::x(4,2),y(4),weight(4),value
  character(64)::arg
  integer::which
  call get_command_argument(1,arg);read(arg,*)which
  x=0;x(:,1)=[.1_dp,.2_dp,.3_dp,.4_dp];x(:,2)=[.4_dp,.1_dp,.2_dp,.3_dp]
  y=[1._dp,2._dp,3._dp,4._dp];weight=1
  sparse=csr_from_dense(x)
  select case(which)
  case(0);model%alpha=-1
  case(1);model%tol=0
  case(2);model%max_iter=-1
  case(3);model%link='invalid'
  case(4);model%power=ieee_value(0._dp,ieee_quiet_nan)
  case(5);model%alpha=ieee_value(0._dp,ieee_positive_inf)
  case(6);model%tol=ieee_value(0._dp,ieee_quiet_nan)
  case(7);weight=0
  case(8);weight(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(9);weight(1)=ieee_value(0._dp,ieee_positive_inf)
  case(10);y(1)=ieee_value(0._dp,ieee_quiet_nan)
  case(11);x(1,1)=ieee_value(0._dp,ieee_positive_inf)
  case(12);model%power=.5_dp;y(1)=-1
  case(13);model%power=1.5_dp;y(1)=-1
  case(14);model%power=2;y(1)=0
  case(15);model%power=3;y(1)=0
  case(16);call model%fit(x,y,weight(:3));stop
  case(17);call model%fit(x,reshape([y,y],[4,2]));stop
  case(18);call model%fit(x,y(:3));stop
  case(19);value=model%score(x,y);stop
  case(20)
    call model%fit(x,y);model%warm_start=.true.;call model%fit(x(:,:1),y);stop
  case(21)
    call model%fit(x,y);value=model%score(x,y,weight(:3));stop
  case(22)
    call model%fit(x,y);weight=0;value=model%score(x,y,weight);stop
  case(23)
    model%power=3;call model%fit(x,y);y(1)=0;value=model%score(x,y);stop
  case(24);model%solver='unknown'
  case(25);sparse%data(1)=ieee_value(0._dp,ieee_quiet_nan);call model%fit(sparse,y);stop
  case(26);call model%fit(sparse,y,weight(:3));stop
  case(27);call model%fit(sparse,reshape([y,y],[4,2]));stop
  case(28);sparse=csr_from_dense(x(:0,:));call model%fit(sparse,y(:0));stop
  case(29)
    call model%fit(sparse,y);sparse=csr_from_dense(x(:,:1));value=model%score(sparse,y);stop
  case(30);call model%fit(sparse,y(:3));stop
  case(31);sparse=csr_from_dense(x(:,:0));call model%fit(sparse,y);stop
  end select
  call model%fit(x,y,weight)
  print *,'unexpected acceptance'
end program
