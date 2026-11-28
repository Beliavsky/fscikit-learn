program test_lda_sparse
  use sklearn_lda, only: latent_dirichlet_allocation
  use sklearn_sparse, only: csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  type(latent_dirichlet_allocation) :: dense,sparse,copy
  type(csr_matrix) :: x,before,query
  real(dp) :: counts(6,4),value
  real(dp), allocatable :: actual(:,:),expected(:,:)
  integer :: i,n,p
  character(32) :: mode
  call get_command_argument(1,mode)
  if(mode=='memory') then
    n=2048; p=50000
    x%n_rows=n; x%n_cols=p
    allocate(x%indptr(n+1),x%indices(2*n),x%data(2*n))
    do i=1,n
      x%indptr(i)=2*i-1
      x%indices(2*i-1:2*i)=[mod(17*i,p)+1,mod(31*i,p)+1]
      x%data(2*i-1:2*i)=[1._dp,2._dp]
    end do
    x%indptr(n+1)=2*n+1
    sparse%n_components=3; sparse%max_iter=2; sparse%max_doc_update_iter=5
    call sparse%fit(x); call sparse%partial_fit(x)
    actual=sparse%transform(x)
    call require(all(ieee_is_finite(actual)),'large CSR LDA finite posterior')
    call require(maxval(abs(sum(actual,dim=2)-1))<1.e-12_dp,'large CSR LDA normalized posterior')
    call require(ieee_is_finite(sparse%score(x)).and.ieee_is_finite(sparse%perplexity(x)), &
      'large CSR LDA finite bound')
    print *, 'Large sparse LDA memory case: PASS'
    stop
  end if
  counts=reshape([0._dp,2._dp,0._dp,1._dp,3._dp,0._dp, &
    0._dp,1._dp,2._dp,0._dp,0._dp,4._dp,0._dp,0._dp,3._dp,2._dp,1._dp,0._dp, &
    0._dp,3._dp,0._dp,1._dp,0._dp,2._dp],[6,4])
  x=csr_from_dense(counts); before=x
  select case(mode)
  case('negative'); x%data(1)=-1
  case('nan'); x%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  case('zero_rows'); x=csr_from_arrays(0,4,[1],[integer::],[real(dp)::])
  case('zero_cols'); x=csr_from_arrays(1,0,[1,1],[integer::],[real(dp)::])
  case('bad_pointer'); x%indptr(1)=0
  case('bad_index'); x%indices(1)=5
  case('unfitted'); actual=sparse%transform(x)
  end select
  sparse%n_components=3; sparse%max_iter=4; sparse%random_state=7; dense=sparse
  actual=sparse%fit_transform(x); expected=dense%fit_transform(counts)
  call require(maxval(abs(actual-expected))<1.e-12_dp,'CSR/dense LDA posterior')
  call require(maxval(abs(sparse%components_-dense%components_))<1.e-12_dp,'CSR/dense LDA components')
  call require(all(x%data==before%data).and.all(x%indices==before%indices).and. &
    all(x%indptr==before%indptr),'CSR LDA preserves storage')
  if(mode=='features') then
    query=csr_from_dense(counts(:,:3)); actual=sparse%transform(query)
  else if(mode=='changed_components') then
    sparse%n_components=4; call sparse%partial_fit(x)
  end if
  call require(abs(sparse%score(x)-dense%score(counts))<1.e-10_dp,'CSR/dense LDA evidence')
  call require(abs(sparse%perplexity(x,.true.)-dense%perplexity(counts,.true.))<1.e-10_dp, &
    'CSR/dense LDA subsampled perplexity')
  copy=sparse; call sparse%partial_fit(x); call copy%partial_fit(x); call dense%partial_fit(counts)
  call require(all(sparse%components_==copy%components_),'CSR LDA copied RNG continuation')
  call require(maxval(abs(sparse%components_-dense%components_))<1.e-8_dp,'CSR/dense LDA streaming')
  call sparse%partial_fit(counts); call copy%partial_fit(x)
  call require(maxval(abs(sparse%components_-copy%components_))<1.e-8_dp,'CSR LDA mixed storage continuation')
  sparse=latent_dirichlet_allocation(); sparse%n_components=2; sparse%learning_method='online'
  sparse%batch_size=2; sparse%max_iter=10; sparse%evaluate_every=1; sparse%perp_tol=1.e9_dp
  dense=sparse; call sparse%fit(x); call dense%fit(counts)
  call require(sparse%n_iter_==dense%n_iter_.and.sparse%n_batch_iter_==dense%n_batch_iter_, &
    'CSR LDA online early-stop counters')
  call require(maxval(abs(sparse%components_-dense%components_))<1.e-12_dp,'CSR LDA online chunks')
  ! Empty stored rows and zero document updates still have a well-defined posterior.
  sparse%max_doc_update_iter=0; actual=sparse%transform(x,.false.)
  call require(all(actual==1),'CSR LDA zero document iterations')
  sparse%max_doc_update_iter=10; query=csr_from_dense(0*counts)
  actual=sparse%transform(query)
  call require(maxval(abs(actual-.5_dp))<1.e-12_dp,'CSR LDA empty document prior')
  value=sparse%perplexity(query)
  call require(.not.ieee_is_finite(value),'CSR LDA zero-word perplexity')
  print *, 'Sparse LDA tests: PASS'
end program
