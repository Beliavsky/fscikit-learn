program test_sgd_one_class_sparse
  use sklearn_sgd_one_class,only:sgd_one_class_svm
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(sgd_one_class_svm)::model,repeat
  type(csr_matrix)::x,original
  real(dp)::one(1,1)
  real(dp),allocatable::data(:),coef(:),scores(:)
  integer,allocatable::ptr(:),indices(:),labels(:),before(:),after(:)
  integer::i,n
  character(32)::mode
  call get_command_argument(1,mode)
  if(trim(mode)=='memory')then
    allocate(ptr(385),indices(768),data(768))
    do i=1,384
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[i,200000];data(2*i-1:2*i)=[.5_dp,1._dp]
    end do
    ptr(385)=769;x=csr_from_arrays(384,200000,ptr,indices,data)
    model%max_iter=3;model%tol=-1;model%average=1
    labels=model%fit_predict(x);call model%partial_fit(x);scores=model%score_samples(x)
    model%warm_start=.true.;call model%fit(x);scores=model%decision_function(x);labels=model%predict(x)
    call require(size(model%coef_)==200000.and.size(labels)==384,'large sparse one-class dimensions')
    print *,'Large CSR SGDOneClassSVM memory case: PASS';stop
  end if
  one=1;x=csr_from_dense(one)
  select case(trim(mode))
  case('unallocated');deallocate(x%data);call model%fit(x)
  case('pointer');x%indptr(2)=1;call model%fit(x)
  case('column');x%indices(1)=2;call model%fit(x)
  case('nan');x%data(1)=ieee_value(0._dp,ieee_quiet_nan);call model%fit(x)
  case('empty');x=csr_from_arrays(0,1,[1],[integer::],[real(dp)::]);call model%fit(x)
  case('zero_width');x=csr_from_arrays(1,0,[1,1],[integer::],[real(dp)::]);call model%fit(x)
  case('weight_count');call model%fit(x,[1._dp,1._dp])
  case('weight_negative');call model%fit(x,[-1._dp])
  case('weight_nan');call model%fit(x,[ieee_value(0._dp,ieee_quiet_nan)])
  case('initial_shape');call model%fit(x,coef_init=[1._dp,1._dp])
  case('partial_width');call model%fit(x);x%n_cols=2;call model%partial_fit(x)
  case('query_width');call model%fit(x);x%n_cols=2;scores=model%score_samples(x)
  case('query_unfitted');labels=model%predict(x)
  case('average_change');call model%partial_fit(x);model%average=1;call model%partial_fit(x)
  end select
  call require(len_trim(mode)==0,'sparse one-class rejection unexpectedly accepted input')
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  model%learning_rate='constant';model%eta0=.1_dp;model%nu=.5_dp;model%max_iter=1;model%tol=-1
  model%shuffle=.false.;model%average=1;original=x;call model%fit(x)
  call require(abs(model%coef_(1)-.1_dp)<1.e-14_dp,'sparse one-class boundary update')
  call require(abs(model%offset_+.0005_dp)<1.e-14_dp,'sparse intercept decay')
  call model%partial_fit(one)
  call require(model%t_==3.and.abs(model%coef_(1)-.09875_dp)<1.e-14_dp,'mixed sparse/dense streaming average')
  call require(all(x%data==original%data).and.all(x%indices==original%indices),'one-class sparse fit preserves storage')
  scores=model%score_samples(x);call require(maxval(abs(scores-model%score_samples(one)))<1.e-14_dp,'mixed query forms')
  x=csr_from_arrays(1,2,[1,4],[2,1,2],[.25_dp,0._dp,.75_dp]);model%fit_intercept=.false.
  call model%fit(x);coef=model%coef_;call model%fit(x)
  call require(all(model%coef_==coef),'one-class duplicate fit repeatability')
  call require(abs(model%coef_(2)-.1_dp)<1.e-14_dp.and.model%coef_(1)==0,'duplicates and explicit zeros are retained')
  repeat=model;repeat%warm_start=.true.;call repeat%fit(x)
  call random_seed(get=after);call require(all(before==after),'sparse one-class preserves intrinsic RNG')
  print *,'CSR SGDOneClassSVM updates, intercept decay, streams and storage: PASS'
end program
