program reference_ridge_estimator_invalid_driver
 use sklearn_common,only:dp
 use sklearn_linear_model,only:ridge
 use sklearn_sparse,only:csr_from_dense
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
 implicit none
 type(ridge)::model
 real(dp)::x(3,2),y(3),w(3)
 real(dp),allocatable::prediction(:),matrix(:,:)
 character(16)::argument
 integer::which
 call get_command_argument(1,argument);read(argument,*)which
 x=reshape([1._dp,2._dp,3._dp,0._dp,1._dp,0._dp],[3,2]);y=[1._dp,3._dp,2._dp];w=1
 select case(which)
 case(1)
  model%solver='unknown'
 case(2)
  model%max_iter=0
 case(3)
  model%tol=-1
 case(4)
  model%tol=ieee_value(1._dp,ieee_quiet_nan)
 case(5)
  model%solver='lbfgs'
 case(6)
  model%positive=.true.;model%solver='lsqr'
 case(7)
  model%alpha=-1
 case(8)
  model%alpha_per_target=[1._dp,2._dp]
 case(9)
  w=ieee_value(1._dp,ieee_quiet_nan)
 case(10)
  call model%fit(x,y,[1._dp]);stop 0
 case(11)
  model%solver='cholesky';call model%fit(csr_from_dense(x),y);stop 0
 case(12)
  model%solver='saga';call model%fit(csr_from_dense(x),y);stop 0
 case(13)
  prediction=model%predict(csr_from_dense(x));stop 0
 case(14)
  call model%fit(x,reshape(y,[3,1]));prediction=model%predict(x);stop 0
 case(15)
  call model%fit(x,y);prediction=model%predict(csr_from_dense(x(:,:1)));stop 0
 case(16)
  model%random_state=-1
 case(17)
  call model%fit(x,y);x(1,1)=ieee_value(1._dp,ieee_quiet_nan)
  matrix=model%predict_matrix(csr_from_dense(x));stop 0
 case(18)
  w=0
 end select
 call model%fit(x,y,w)
end program
