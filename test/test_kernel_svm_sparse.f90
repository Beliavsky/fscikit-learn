program test_kernel_svm_sparse
  use sklearn_common,only:dp,require
  use sklearn_kernel_svm,only:svr,nu_svr,one_class_svm
  use sklearn_sparse,only:csr_matrix,csr_from_arrays,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(svr)::s,d
  type(nu_svr)::v
  type(one_class_svm)::o
  type(csr_matrix)::x,q,original
  real(dp)::a(8,3),y(8),w(8)
  real(dp),allocatable::p(:),expected(:),targets(:),coefficient(:,:)
  integer,allocatable::ptr(:),indices(:),labels(:)
  real(dp),allocatable::data(:)
  integer::i
  character(32)::mode
  call get_command_argument(1,mode)
  if(trim(mode)=='memory'.or.trim(mode)=='memory_linear')then
    allocate(ptr(65),indices(128),data(128),targets(64))
    do i=1,64
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[i,2000001-i]
      data(2*i-1:2*i)=[real(i,dp)/64,1._dp];targets(i)=sin(real(i,dp))
    end do
    ptr(65)=129;x=csr_from_arrays(64,200000000,ptr,indices,data)
    s%gamma=.7_dp;v%gamma=.7_dp;o%gamma=.7_dp
    if(trim(mode)=='memory_linear')then
      s%kernel='linear';v%kernel='linear';o%kernel='linear'
    end if
    call s%fit(x,targets);call v%fit(x,targets);labels=o%fit_predict(x)
    call require(.not.allocated(s%x_fit_).and..not.allocated(v%x_fit_).and..not.allocated(o%x_fit_), &
      'large kernel SVM fits never allocate dense designs')
    call require(.not.allocated(s%support_vectors_).and..not.allocated(v%support_vectors_).and. &
      .not.allocated(o%support_vectors_),'large sparse SVM support vectors stay sparse')
    if(trim(mode)=='memory_linear')then
      call require(.not.allocated(s%coef_).and..not.allocated(v%coef_).and..not.allocated(o%coef_), &
        'linear CSR coefficients never allocate dense feature vectors')
      call require(size(s%coef_csr_%data)<=128.and.size(v%coef_csr_%data)<=128.and. &
        size(o%coef_csr_%data)<=128,'linear CSR coefficients bounded by stored nodes')
      call require(s%coef_csr_%n_cols==200000000,'linear CSR coefficient width')
    end if
    p=s%predict(x);p=v%predict(x);p=o%score_samples(x)
    call require(size(p)==64.and.size(labels)==64,'large sparse kernel SVM queries')
    print *,'Large CSR kernel SVM memory case: PASS';stop
  end if
  a=reshape([(sin(real(i,dp))+.8_dp,i=1,24)],shape(a));y=sin(a(:,1));w=1;w(1)=0
  x=csr_from_dense(a);q=x;original=x
  select case(trim(mode))
  case('unfitted');p=s%predict(x)
  case('nan');x%data(1)=ieee_value(0._dp,ieee_quiet_nan);call s%fit(x,y)
  case('inf');x%data(1)=ieee_value(0._dp,ieee_positive_inf);call s%fit(x,y)
  case('shape');call s%fit(x,y);q%n_cols=4;p=s%predict(q)
  case('precomputed');s%kernel='precomputed';call s%fit(x,y)
  case('dense_fit_csr_query');call s%fit(a,y);p=s%predict(x)
  case('bad_index');x%indices(1)=0;call s%fit(x,y)
  case('bad_ptr');x%indptr(1)=0;call s%fit(x,y)
  case('negative_weight');w(2)=-1;call s%fit(x,y,w)
  case('bad_y');call s%fit(x,y(:7))
  case('bad_nu');v%nu=0;call v%fit(x,y)
  case('bad_epsilon');s%epsilon=-1;call s%fit(x,y)
  end select
  call require(len_trim(mode)==0,'guard unexpectedly accepted invalid input')
  s%tol=1.e-8_dp;d%tol=1.e-8_dp
  call s%fit(x,y,w);call d%fit(a,y,w)
  p=s%predict(x);expected=d%predict(a)
  call require(maxval(abs(p-expected))<1.e-5_dp,'CSR and dense SVR agree')
  expected=s%predict(a);call require(maxval(abs(p-expected))<1.e-12_dp,'dense queries after sparse SVR fit')
  call require(abs(s%score(x,y)-s%score(a,y))<1.e-12_dp,'CSR regression score')
  call require(all(x%data==original%data).and.all(x%indices==original%indices), 'CSR fit preserves caller storage')
  call require(s%sparse_fit_.and..not.allocated(s%x_fit_).and..not.allocated(s%support_vectors_), &
    'CSR SVR stores sparse design and support vectors')
  call s%support_vectors_csr_%validate()
  call v%fit(x,y,w);p=v%predict(x);expected=v%decision_function(a)
  call require(maxval(abs(p-expected))<1.e-12_dp,'CSR NuSVR prediction and decisions')
  labels=o%fit_predict(x,w);p=o%score_samples(x);expected=o%decision_function(x)
  call require(maxval(abs(p+o%intercept_-expected))<1.e-12_dp,'CSR one-class raw and offset scores')
  call require(all(labels==o%predict(a)),'CSR one-class labels and dense queries')
  call s%fit(a,y)
  call require(.not.s%sparse_fit_.and..not.allocated(s%x_fit_csr_).and. &
    .not.allocated(s%support_vectors_csr_),'dense refit clears sparse fitted state')
  s%kernel='linear';call s%fit(x,y)
  call require(s%coef_csr_%n_cols==3.and.s%coef_csr_%n_rows==1,'CSR linear coefficient shape')
  call require(.not.allocated(s%coef_),'CSR fit does not expose dense coefficients')
  d%kernel='linear';call d%fit(a,y);coefficient=s%coef_csr_%to_dense()
  call require(maxval(abs(coefficient(1,:)-d%coef_))<1.e-5_dp,'dense and sparse linear coefficients agree')
  call s%fit(a,y);call require(.not.allocated(s%coef_csr_),'dense refit clears CSR coefficients')
  s%kernel='rbf';call s%fit(x,y);call require(.not.allocated(s%coef_).and..not.allocated(s%coef_csr_), &
    'nonlinear refit clears linear coefficients')
  x=csr_from_arrays(3,5,[1,1,1,1],[integer::],[real(dp)::])
  call s%fit(x,[1._dp,1._dp,1._dp]);p=s%predict(x)
  call require(s%gamma_==1.and.all(abs(p-1)<1.e-14_dp),'empty sparse rows and constant targets')
  s%kernel='linear';call s%fit(x,[1._dp,1._dp,1._dp])
  call require(size(s%coef_csr_%data)==0.and.all(s%coef_csr_%indptr==1), &
    'zero linear coefficients use empty canonical CSR storage')
  v%kernel='linear';o%kernel='linear'
  call v%fit(x,[1._dp,1._dp,1._dp]);call o%fit(x)
  call require(size(v%coef_csr_%data)==0.and.size(o%coef_csr_%data)==0,'empty CSR coefficients for all families')
  v%kernel='rbf';o%kernel='rbf'
  call v%fit(x,[1._dp,1._dp,1._dp]);call o%fit(x)
  call require(.not.allocated(v%coef_csr_).and..not.allocated(o%coef_csr_),'nonlinear refit clears all CSR coefficients')
  print *,'CSR SVR, NuSVR and OneClassSVM fits, queries, scores and state: PASS'
end program
