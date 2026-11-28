program test_svr_libsvm
  use sklearn_kernel_svm,only:svr
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use sklearn_common,only:dp,require
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_finite
  implicit none
  type(svr)::model
  type(csr_matrix)::sparse
  real(dp)::x(2,1),y(2),w(2),asym(2,2),bad(2,1)
  real(dp),allocatable::v(:),before(:)
  character(32)::mode
  x(:,1)=[-1._dp,1._dp];y=[-2._dp,2._dp];w=[1._dp,.25_dp];model%kernel='linear'
  call get_command_argument(1,mode)
  select case(trim(mode))
  case('epsilon_negative');model%epsilon=-1;call model%fit(x,y)
  case('epsilon_nan');model%epsilon=ieee_value(0._dp,ieee_quiet_nan);call model%fit(x,y)
  case('c_zero');model%c=0;call model%fit(x,y)
  case('c_nan');model%c=ieee_value(0._dp,ieee_quiet_nan);call model%fit(x,y)
  case('cap_zero');model%max_iter=0;call model%fit(x,y)
  case('tol_zero');model%tol=0;call model%fit(x,y)
  case('weight_count');call model%fit(x,y,[1._dp])
  case('weight_zero');call model%fit(x,y,[0._dp,0._dp])
  case('weight_negative');call model%fit(x,y,[1._dp,-1._dp])
  case('weight_nan');call model%fit(x,y,[1._dp,ieee_value(0._dp,ieee_quiet_nan)])
  case('target_count');call model%fit(x,y(:1))
  case('fit_nan');bad=x;bad(1,1)=ieee_value(0._dp,ieee_quiet_nan);call model%fit(bad,y)
  case('query_width');call model%fit(x,y);v=model%predict(reshape([1._dp,2._dp],[1,2]))
  case('unfitted');v=model%predict(x)
  case('kernel');model%kernel='bad';call model%fit(x,y)
  end select
  call require(len_trim(mode)==0,'SVR guard unexpectedly accepted input')
  call require(model%shrinking,'SVR defaults to Source shrinking')
  model%tol=1.e-9_dp;call model%fit(x,y,w)
  call require(abs(model%coef_(1)-.5_dp)<1.e-12_dp.and.abs(model%intercept_+1.4_dp)<1.e-12_dp,'SVR unequal-box Source solution')
  model%shrinking=.false.;call model%fit(x,y,w);before=model%predict(x)
  sparse=csr_from_dense(x);call model%fit(sparse,y,w);v=model%predict(sparse)
  call require(maxval(abs(v-before))<1.e-12_dp,'SVR dense/CSR weighted agreement')
  call require(abs(model%score(sparse,y,w)-model%score(x,y,w))<1.e-12_dp,'weighted dense/CSR scoring')
  asym(1,:)=[2._dp,5._dp];asym(2,:)=[-1._dp,3._dp];model%kernel='precomputed';y=[-1._dp,1._dp]
  call model%fit(asym,y);v=model%predict(asym)
  call require(all(abs(model%dual_coef_-[-1._dp,1._dp])<1.e-12_dp).and.abs(model%intercept_-2.5_dp)<1.e-12_dp, &
    'SVR asymmetric Source cache coefficients')
  call require(all(abs(v-[5.5_dp,6.5_dp])<1.e-12_dp).and.model%n_iter_==6,'SVR asymmetric Source predictions/count')
  model%kernel='sigmoid';model%coef0=-.2_dp;model%gamma=.4_dp;call model%fit(x,y)
  v=model%predict(x);call require(all(ieee_is_finite(v)),'SVR signed sigmoid kernel')
  model%kernel='rbf';model%gamma=0;model%epsilon=100;call model%fit(x,y)
  v=model%predict(x);call require(model%n_support_==0.and.all(v==0),'SVR empty-support Source threshold')
  print *,'LIBSVM epsilon-SVR boxes, shrinking, sigmoid, asymmetric kernels and weighted score: PASS'
end program
