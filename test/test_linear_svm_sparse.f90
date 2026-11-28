program test_linear_svm_sparse
  use sklearn_common,only:dp,require
  use sklearn_linear_svm,only:linear_svc,linear_svr
  use sklearn_sparse,only:csr_matrix,csr_from_arrays,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(linear_svc)::s,d
  type(linear_svr)::r
  type(csr_matrix)::x,q,original
  real(dp)::pair(2,1),w(2)
  real(dp),allocatable::z(:),targets(:),data(:),scores(:,:)
  integer,allocatable::ptr(:),indices(:),labels(:),predicted(:)
  integer::i
  character(32)::mode
  call get_command_argument(1,mode)
  if(trim(mode)=='memory')then
    allocate(ptr(129),indices(256),data(256),targets(128),labels(128))
    do i=1,128
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[1,i+1]
      data(2*i-1:2*i)=[real(mod(i,3)-1,dp),1._dp]
      targets(i)=sin(real(i,dp));labels(i)=mod(i,3)
    end do
    ptr(129)=257;x=csr_from_arrays(128,200000,ptr,indices,data)
    call s%fit(x,labels);call r%fit(x,targets)
    predicted=s%predict(x);z=r%predict(x)
    call require(size(s%coef_,2)==200000.and.size(r%coef_)==200000,'large CSR linear SVM coefficient dimensions')
    call require(s%converged_.and.r%converged_,'large CSR linear SVM solvers converge')
    call require(size(predicted)==128.and.size(z)==128,'large CSR linear SVM predictions')
    print *,'Large CSR linear SVM memory case: PASS';stop
  end if
  pair(:,1)=[-1._dp,1._dp];x=csr_from_dense(pair);q=x
  select case(trim(mode))
  case('unfitted');predicted=s%predict(x)
  case('nan');x%data(1)=ieee_value(0._dp,ieee_quiet_nan);call s%fit(x,[-3,9])
  case('inf');x%data(1)=ieee_value(0._dp,ieee_positive_inf);call r%fit(x,[-2._dp,2._dp])
  case('shape');call s%fit(x,[-3,9]);q%n_cols=2;predicted=s%predict(q)
  case('bad_index');x%indices(1)=0;call s%fit(x,[-3,9])
  case('bad_ptr');x%indptr(1)=0;call s%fit(x,[-3,9])
  case('bad_y');call s%fit(x,[-3])
  case('one_class');call s%fit(x,[-3,-3])
  case('negative_weight');call s%fit(x,[-3,9],sample_weight=[-1._dp,1._dp])
  case('bad_c');s%c=0;call s%fit(x,[-3,9])
  case('bad_epsilon');r%epsilon=-1;call r%fit(x,[-2._dp,2._dp])
  case('bad_class_weight');call s%fit(x,[-3,9],class_weight=[1._dp])
  case('conflicting_weights');s%class_weight_balanced=.true.;call s%fit(x,[-3,9],class_weight=[1._dp,1._dp])
  end select
  call require(len_trim(mode)==0,'linear SVM guard unexpectedly accepted input')
  s%fit_intercept=.false.;s%tol=1.e-12_dp;call s%fit(x,[-3,9])
  call require(abs(s%coef_(1,1)-.8_dp)<1.e-13_dp,'CSR analytic squared-hinge optimum')
  call require(abs(s%objective_(1)-.4_dp)<1.e-13_dp.and.s%converged_,'CSR classification objective')
  r%fit_intercept=.false.;r%tol=1.e-12_dp;r%epsilon=.5_dp;call r%fit(x,[-2._dp,2._dp])
  call require(abs(r%coef_(1)-1.2_dp)<1.e-13_dp.and.abs(r%objective_-.9_dp)<1.e-13_dp, &
    'CSR analytic squared epsilon-insensitive optimum')
  x=csr_from_arrays(2,1,[1,3,5],[1,1,1,1],[-.25_dp,-.75_dp,.25_dp,.75_dp]);original=x
  call s%fit(x,[-3,9]);call require(abs(s%coef_(1,1)-.8_dp)<1.e-13_dp,'duplicate CSR entries share a feature')
  call require(all(x%data==original%data).and.all(x%indices==original%indices),'CSR linear fit preserves input')
  predicted=s%predict(pair);call require(all(predicted==s%predict(x)),'dense query after CSR fit')
  call d%fit(pair,[-3,9]);predicted=d%predict(x)
  call require(all(predicted==d%predict(pair)),'CSR query after dense fit')
  call require(s%score(x,[-3,9])==1,'CSR linear classification score')
  z=r%predict(pair);call require(maxval(abs(z-r%predict(x)))<1.e-13_dp,'mixed linear regression input formats')
  call require(abs(r%score(x,[-2._dp,2._dp])-r%score(pair,[-2._dp,2._dp]))<1.e-13_dp,'CSR regression score')
  x=csr_from_arrays(2,3,[1,1,1],[integer::],[real(dp)::]);w=[1._dp,3._dp]
  s%fit_intercept=.true.;s%intercept_scaling=2;call s%fit(x,[-3,9],sample_weight=w)
  call require(abs(s%intercept_(1)-16._dp/33)<1.e-13_dp,'CSR regularized scaled synthetic intercept')
  x=csr_from_arrays(3,3,[1,1,1,1],[integer::],[real(dp)::]);s%fit_intercept=.false.
  call s%fit(x,[-4,7,21]);predicted=s%predict(x)
  call require(all(predicted==-4).and.s%n_iter_==0,'CSR multiclass stationary ties')
  r%epsilon=100;call r%fit(x,[1._dp,2._dp,3._dp]);z=r%predict(x)
  call require(all(z==0).and.r%converged_.and.r%n_iter_==0,'CSR regression inactive loss tube')
  call s%fit(pair,[-3,9]);call require(all(shape(s%coef_)==[1,1]),'dense refit resizes multiclass coefficients')
  print *,'CSR linear SVC/SVR objectives, weights, duplicates and mixed queries: PASS'
end program
