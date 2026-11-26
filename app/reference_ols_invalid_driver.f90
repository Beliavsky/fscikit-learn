program reference_ols_invalid_driver
 use sklearn_common,only:dp
 use sklearn_linear_model,only:linear_regression,huber_regressor
 use sklearn_sparse,only:csr_from_dense
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
 implicit none
 type(linear_regression)::model
 type(huber_regressor)::huber
 real(dp)::x(3,2),y(3),w(3),targets(3,2)
 real(dp),allocatable::pred(:),matrix(:,:)
 integer::which
 character(16)::argument
 call get_command_argument(1,argument);read(argument,*)which
 x=reshape([1._dp,2._dp,3._dp,0._dp,1._dp,0._dp],[3,2]);y=[1._dp,3._dp,2._dp];w=1
 targets(:,1)=y;targets(:,2)=-y
 select case(which)
 case(1)
  call model%fit(x,y,[1._dp]);stop 0
 case(2)
  w=-1
 case(3)
  w=ieee_value(1._dp,ieee_quiet_nan)
 case(4)
  w=0
 case(5)
  call model%fit(x,y(:2));stop 0
 case(6)
  y=ieee_value(1._dp,ieee_quiet_nan)
 case(7)
  x(1,1)=ieee_value(1._dp,ieee_quiet_nan)
 case(8)
  call model%fit(x(:0,:),y(:0));stop 0
 case(9)
  model%positive=.true.;call model%fit(csr_from_dense(x),y);stop 0
 case(10)
  pred=model%predict(csr_from_dense(x));stop 0
 case(11)
  call model%fit(x,targets);pred=model%predict(x);stop 0
 case(12)
  call model%fit(x,y);pred=model%predict(csr_from_dense(x(:,:1)));stop 0
 case(13)
  call model%fit(x,targets);x(1,1)=ieee_value(1._dp,ieee_quiet_nan)
  matrix=model%predict_matrix(csr_from_dense(x));stop 0
 case(14)
  call model%fit(x,targets(:,:0));stop 0
 case(15)
  call model%fit(x,y,sample_weight=ieee_value(1._dp,ieee_quiet_nan));stop 0
 case(16)
  call model%fit(csr_from_dense(x(:,:0)),y);stop 0
 case(17)
  call huber%fit(x,targets);stop 0
 end select
 call model%fit(x,y,w)
end program
