program test_nusvc_sparse
  use sklearn_nu_classification,only:nu_svc
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_common,only:dp,require
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_finite
  implicit none
  type(nu_svc)::model
  type(csr_matrix)::x,original
  real(dp)::pair(2,1),query(1,1)
  real(dp),allocatable::scores(:,:),data(:),linear(:,:)
  integer,allocatable::ptr(:),indices(:),labels(:),target(:),before(:),after(:)
  integer::i,n
  character(32)::mode
  call get_command_argument(1,mode)
  if(trim(mode)=='memory')then
    allocate(ptr(129),indices(256),data(256),target(128))
    do i=1,128
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[i,200000];data(2*i-1:2*i)=[.5_dp,1._dp];target(i)=mod(i,3)
    end do
    ptr(129)=257;x=csr_from_arrays(128,200000,ptr,indices,data)
    call model%fit(x,target);scores=model%decision_function(x);labels=model%predict(x)
    model%kernel='linear';call model%fit(x,target);scores=model%decision_function(x);labels=model%predict(x)
    call require(.not.allocated(model%support_vectors_).and..not.allocated(model%coef_), &
      'wide CSR NuSVC has no dense feature storage')
    call require(allocated(model%coef_csr_).and.model%n_features_in_==200000,'wide sparse linear coefficients')
    call require(all(ieee_is_finite(scores)).and.size(labels)==128,'wide CSR NuSVC query dimensions')
    print *,'Large CSR NuSVC memory case: PASS';stop
  end if
  pair(:,1)=[-1._dp,1._dp];query=0;x=csr_from_dense(pair);model%kernel='linear';model%tol=1.e-9_dp
  select case(trim(mode))
  case('unallocated');deallocate(x%data);call model%fit(x,[-3,9])
  case('pointer');x%indptr(3)=1;call model%fit(x,[-3,9])
  case('column');x%indices(1)=2;call model%fit(x,[-3,9])
  case('nan');x%data(1)=ieee_value(0._dp,ieee_quiet_nan);call model%fit(x,[-3,9])
  case('empty');x=csr_from_arrays(0,1,[1],[integer::],[real(dp)::]);call model%fit(x,[integer::])
  case('zero_width');x=csr_from_arrays(2,0,[1,1,1],[integer::],[real(dp)::]);call model%fit(x,[-3,9])
  case('target_count');call model%fit(x,[-3])
  case('weight_count');call model%fit(x,[-3,9],[1._dp])
  case('precomputed');model%kernel='precomputed';call model%fit(x,[-3,9])
  case('kernel');model%kernel='bad';call model%fit(x,[-3,9])
  case('unfitted');labels=model%predict(x)
  case('dense_fit_query');call model%fit(pair,[-3,9]);labels=model%predict(x)
  case('query_width');call model%fit(x,[-3,9]);x%n_cols=2;labels=model%predict(x)
  case('query_nan');call model%fit(x,[-3,9]);x%data(1)=ieee_value(0._dp,ieee_quiet_nan);labels=model%predict(x)
  end select
  call require(len_trim(mode)==0,'CSR NuSVC rejection unexpectedly accepted input')
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  original=x;call model%fit(x,[-3,9],[1._dp,.5_dp])
  linear=model%coef_csr_%to_dense()
  call require(abs(linear(1,1)-1._dp)<1.e-12_dp.and.abs(model%intercept_(1))<1.e-12_dp,'weighted CSR optimum')
  call require(all(x%data==original%data).and.all(x%indices==original%indices),'CSR NuSVC preserves caller storage')
  call require(model%sparse_fit_.and..not.allocated(model%coef_),'CSR NuSVC storage convention')
  scores=model%decision_function(query);labels=model%predict(query)
  call require(abs(scores(1,1))<1.e-12_dp.and.labels(1)==9,'dense queries after CSR fit')
  call require(model%score(x,[-3,9],[1._dp,.5_dp])> .9_dp,'weighted CSR score')
  model%kernel='rbf';call model%fit(x,[-3,9]);call require(.not.allocated(model%coef_csr_),'nonlinear refit clears sparse coefficients')
  call model%fit(pair,[-3,9]);call require(.not.model%sparse_fit_.and..not.allocated(model%x_fit_csr_), 'dense refit clears CSR training storage')
  call require(.not.allocated(model%support_vectors_csr_),'dense refit clears sparse supports')
  x=csr_from_arrays(3,2,[1,4,7,7],[2,1,2,2,1,2],[.25_dp,-1._dp,.75_dp,.5_dp,1._dp,1.5_dp]);original=x
  model%kernel='linear';call model%fit(x,[-3,9,-3],[1._dp,1._dp,0._dp])
  call require(all(x%data==original%data).and.all(x%indices==original%indices),'sorting and duplicates preserve original')
  call require(all(model%support_vectors_csr_%indices(1:3)==[1,2,2]),'owned support sorting retains duplicate nodes')
  call require(model%coef_csr_%n_cols==2,'sparse linear width on refit')
  call random_seed(get=after);call require(all(before==after),'CSR NuSVC preserves intrinsic RNG')
  print *,'CSR NuSVC storage, weights, dense queries, duplicates and refits: PASS'
end program
