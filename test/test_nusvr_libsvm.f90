program test_nusvr_libsvm
  use sklearn_kernel_svm,only:nu_svr
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_finite
  implicit none
  type(nu_svr)::model
  type(csr_matrix)::sparse
  real(dp)::x(2,1),y(2),w(2),asym(2,2),bad(2,1)
  real(dp),allocatable::v(:),before(:)
  character(32)::mode
  x(:,1)=[-1._dp,1._dp];y=[-2._dp,2._dp];w=[1._dp,.25_dp];model%kernel='linear'
  call get_command_argument(1,mode)
  select case(trim(mode))
  case('nu_zero');model%nu=0;call model%fit(x,y)
  case('nu_high');model%nu=2;call model%fit(x,y)
  case('nu_nan');model%nu=ieee_value(0._dp,ieee_quiet_nan);call model%fit(x,y)
  case('c_zero');model%c=0;call model%fit(x,y)
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
  call require(len_trim(mode)==0,'NuSVR guard unexpectedly accepted input')
  call require(model%shrinking,'NuSVR defaults to Source shrinking')
  model%tol=1.e-9_dp;call model%fit(x,y)
  call require(abs(model%coef_(1)-1._dp)<1.e-12_dp.and.abs(model%intercept_)<1.e-12_dp.and.abs(model%epsilon_-1._dp)<1.e-12_dp, &
    'NuSVR two-point Source solution and learned tube')
  call require(all(abs(sum(model%dual_variables_,dim=1)-.5_dp)<1.e-12_dp),'NuSVR independent group masses')
  call require(model%epsilon_==max(0._dp,model%epsilon_signed_),'NuSVR signed/clipped tube diagnostic')
  model%nu=1;call model%fit(x,y);v=model%predict(x)
  call require(all(abs(v-y)<1.e-12_dp).and.abs(model%epsilon_)<1.e-12_dp,'NuSVR nu=1 zero-width tube')
  model%nu=.5_dp;model%shrinking=.false.;call model%fit(x,y,w);before=model%predict(x)
  sparse=csr_from_dense(x);call model%fit(sparse,y,w);v=model%predict(sparse)
  call require(maxval(abs(v-before))<1.e-12_dp,'NuSVR weighted dense/CSR agreement')
  call require(all(abs(sum(model%dual_variables_,dim=1)-.3125_dp)<1.e-12_dp),'NuSVR weighted mass initialization')
  call require(abs(model%score(sparse,y,w)-model%score(x,y,w))<1.e-12_dp,'NuSVR weighted score forwarding')
  asym(1,:)=[2._dp,5._dp];asym(2,:)=[-1._dp,3._dp];model%kernel='precomputed';y=[-1._dp,1._dp]
  call model%fit(asym,y);v=model%predict(asym)
  call require(all(abs(model%dual_coef_-[-.5_dp,.5_dp])<1.e-12_dp).and.abs(model%intercept_-1.25_dp)<1.e-12_dp, &
    'NuSVR asymmetric Source coefficients')
  call require(all(abs(v-[2.75_dp,3.25_dp])<1.e-12_dp),'NuSVR asymmetric Source row evaluation')
  model%kernel='sigmoid';model%coef0=-.2_dp;model%gamma=.4_dp;call model%fit(x,y)
  v=model%predict(x);call require(all(ieee_is_finite(v)),'NuSVR signed sigmoid kernel')
  model%kernel='rbf';model%gamma=0;y=3;call model%fit(x,y);v=model%predict(x)
  call require(model%n_support_==0.and.all(v==3).and.model%epsilon_==0,'NuSVR constant target with zero margin scale')
  print *,'LIBSVM NuSVR groups, shrinking, learned tube, sigmoid and asymmetric kernels: PASS'
end program
