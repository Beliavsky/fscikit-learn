program reference_kernel_ridge_invalid_driver
  use sklearn_common,only:dp
  use sklearn_kernel_ridge,only:kernel_ridge
  use sklearn_sparse,only:csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(kernel_ridge)::model
  real(dp)::x(3,2),y(3),w(3),targets(3,2)
  real(dp),allocatable::pred(:)
  integer::which
  character(20)::argument
  call get_command_argument(1,argument);read(argument,*)which
  x=1;y=1;w=1;targets=1
  select case(which)
  case(1);pred=model%predict(x);stop
  case(2);model%alpha=-1
  case(3);allocate(model%alpha_values(0))
  case(4);model%alpha_values=[1._dp,2._dp]
  case(5);model%alpha=ieee_value(0._dp,ieee_quiet_nan)
  case(6);call model%fit(x,y(:2));stop
  case(7);call model%fit(x(:,:0),y);stop
  case(8);x=ieee_value(x,ieee_quiet_nan)
  case(9);y=ieee_value(y,ieee_quiet_nan)
  case(10);call model%fit(x,y,w(:2));stop
  case(11);w(1)=-1;call model%fit(x,y,w);stop
  case(12);w=ieee_value(w,ieee_quiet_nan);call model%fit(x,y,w);stop
  case(13);call model%fit(x,y,-1._dp);stop
  case(14);model%gamma=-2
  case(15);model%gamma=ieee_value(0._dp,ieee_quiet_nan)
  case(16);model%degree=-1
  case(17);model%kernel='callable'
  case(18);model%kernel='gp'
  case(19);model%kernel='chi2'
  case(20);model%kernel='precomputed'
  case(21);model%kernel='precomputed';call model%fit(csr_from_dense(x),y);stop
  case(22);model%kernel='gp';call model%fit(csr_from_dense(x),y);stop
  case(23);model%kernel='chi2';model%gamma=1;call model%fit(csr_from_dense(x),y);stop
  case(24);call model%fit(x,y);pred=model%predict(x(:,:1));stop
  case(25)
    model%kernel='precomputed';call model%fit(spread([1._dp,1._dp,1._dp],2,3),y)
    pred=model%predict(x);stop
  case(26);call model%fit(x,targets);pred=model%predict(x);stop
  case(27)
    x=0;model%alpha_values=[0._dp,1._dp];call model%fit(x,targets);stop
  case(28);model%kernel='unknown'
  case(29);x=ieee_value(x,ieee_quiet_nan);call model%fit(csr_from_dense(x),y);stop
  case(30);model%kernel='callable';model%kernel_function=>bad_kernel
  end select
  call model%fit(x,y)
contains
  real(dp)function bad_kernel(a,b)result(value)
    real(dp),intent(in)::a(:),b(:)
    value=ieee_value(0._dp,ieee_quiet_nan)
  end function
end program
