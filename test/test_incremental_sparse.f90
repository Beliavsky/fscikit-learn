program test_incremental_sparse
  use sklearn_common,only:dp,require
  use sklearn_incremental_pca,only:incremental_pca
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(incremental_pca)::dense,sparse
  type(csr_matrix)::s,q
  real(dp)::x(37,5)
  real(dp),allocatable::a(:,:),b(:,:),wide(:,:)
  integer::i
  character(20)::arg
  call get_command_argument(1,arg)
  do i=1,37
    x(i,:)=[sin(real(i,dp)),cos(.3_dp*i),real(mod(i,4),dp),sin(.7_dp*i),.03_dp*i]
  end do
  dense%n_components=2;dense%batch_size=7;dense%whiten=.true.;sparse=dense
  s=csr_from_dense(x,store_zeros=.true.)
  a=dense%fit_transform(x);b=sparse%fit_transform(s)
  call require(maxval(abs(a-b))<1.e-10_dp,'IncrementalPCA CSR batch fit')
  call require(maxval(abs(s%to_dense()-x))==0,'IncrementalPCA preserves CSR input')
  a=dense%transform(s);b=sparse%transform(x)
  call require(maxval(abs(a-b))<1.e-10_dp,'IncrementalPCA mixed query storage')
  call sparse%fit(x(:30,:));call sparse%fit(s)
  call require(sparse%n_samples_seen_==37.and.sparse%batch_size_==7,'IncrementalPCA sparse refit resets state')
  if(arg=='--memory')then
    ! Keep the training SVD small; exercise a tall query with bounded batches.
    allocate(wide(37,50))
    do i=1,50
      wide(:,i)=x(:,mod(i-1,5)+1)*sin(real(i,dp))
    end do
    dense%n_components=2;dense%batch_size=256;call dense%fit(wide)
    q%n_rows=1000000;q%n_cols=50
    allocate(q%indptr(q%n_rows+1),q%indices(q%n_rows),q%data(q%n_rows))
    q%indptr=[(i,i=1,q%n_rows+1)];q%indices=1;q%data=.3_dp
    b=dense%transform(q)
    call require(all(shape(b)==[q%n_rows,2]),'IncrementalPCA tall CSR query shape')
    call require(maxval(abs(b(1,:)-b(q%n_rows,:)))==0,'IncrementalPCA tall CSR query values')
    print *,'IncrementalPCA CSR 1000000 by 50 memory: PASS'
  end if
  print *,'IncrementalPCA CSR batching, mixed queries and refits: PASS'
end program
