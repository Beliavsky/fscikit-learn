program test_oneclass_libsvm
  use sklearn_kernel_svm,only:one_class_svm
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_finite
  implicit none
  type(one_class_svm)::model
  type(csr_matrix)::sparse
  real(dp)::x(4,2),asym(2,2),bad(4,2)
  real(dp),allocatable::scores(:),before(:)
  integer,allocatable::labels(:)
  character(32)::mode
  x(1,:)=[1._dp,2._dp];x(2,:)=[2._dp,1._dp];x(3,:)=[3._dp,2._dp];x(4,:)=[2._dp,3._dp]
  asym(1,:)=[2._dp,5._dp];asym(2,:)=[-1._dp,3._dp]
  call get_command_argument(1,mode)
  select case(trim(mode))
  case('nu_zero');model%nu=0;call model%fit(x)
  case('nu_one');model%nu=1;call model%fit(x)
  case('nu_nan');model%nu=ieee_value(0._dp,ieee_quiet_nan);call model%fit(x)
  case('cap_zero');model%max_iter=0;call model%fit(x)
  case('tol_zero');model%tol=0;call model%fit(x)
  case('weight_count');call model%fit(x,[1._dp])
  case('weight_zero');call model%fit(x,[0._dp,0._dp,0._dp,0._dp])
  case('weight_negative');call model%fit(x,[1._dp,-1._dp,1._dp,1._dp])
  case('weight_nan');call model%fit(x,[1._dp,ieee_value(0._dp,ieee_quiet_nan),1._dp,1._dp])
  case('fit_nan');bad=x;bad(1,1)=ieee_value(0._dp,ieee_quiet_nan);call model%fit(bad)
  case('query_width');call model%fit(x);scores=model%decision_function(x(:,:1))
  case('query_nan');call model%fit(x);bad=x;bad(1,1)=ieee_value(0._dp,ieee_quiet_nan);scores=model%decision_function(bad)
  case('unfitted');scores=model%decision_function(x)
  case('kernel');model%kernel='bad';call model%fit(x)
  end select
  call require(len_trim(mode)==0,'OneClassSVM guard unexpectedly accepted input')
  call require(model%shrinking,'OneClassSVM defaults to Source shrinking')
  model%kernel='precomputed';call model%fit(asym);scores=model%decision_function(asym)
  call require(model%n_iter_==0.and.model%fit_status_==0,'asymmetric one-class cache stop status')
  call require(all(model%support_==[1]).and.abs(model%dual_coef_(1)-1._dp)<1.e-12_dp,'one-class weighted prefix initial solution')
  call require(abs(model%offset_-3.5_dp)<1.e-12_dp.and.all(abs(scores-[-1.5_dp,-4.5_dp])<1.e-12_dp), &
    'one-class Source asymmetric row-cache threshold')
  model%kernel='sigmoid';model%gamma=.4_dp;model%coef0=-.2_dp;call model%fit(x)
  scores=model%decision_function(x);call require(all(ieee_is_finite(scores)),'signed sigmoid coefficient support')
  model%shrinking=.false.;call model%fit(x);before=model%decision_function(x)
  sparse=csr_from_dense(x);call model%fit(sparse);scores=model%decision_function(sparse)
  call require(maxval(abs(scores-before))<1.e-8_dp,'dense/CSR sigmoid one-class agreement')
  labels=model%fit_predict(sparse);call require(all(labels==model%predict(sparse)),'CSR fit_predict identity')
  model%kernel='rbf';model%gamma=0;call model%fit(x);labels=model%predict(x)
  call require(all(labels==-1).and.model%n_iter_==0,'constant-cache zero-margin prediction convention')
  print *,'LIBSVM OneClassSVM prefix, shrinking, sigmoid, asymmetric cache and zero margins: PASS'
end program
