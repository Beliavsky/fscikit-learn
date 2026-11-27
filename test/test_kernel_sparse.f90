program test_kernel_sparse
  use sklearn_kernel_pca,only:kernel_pca
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_common,only:dp,require
  implicit none
  type(kernel_pca)::dense,sparse
  type(csr_matrix)::s,before,q
  real(dp)::x(12,3)
  real(dp),allocatable::a(:,:),b(:,:)
  integer::i
  character(20)::arg
  call get_command_argument(1,arg)
  do i=1,12
    x(i,:)=[sin(real(i,dp)),cos(.7_dp*i),real(mod(i,3),dp)]
  end do
  s=csr_from_dense(x,store_zeros=.true.);before=s
  dense%kernel='rbf';dense%gamma=.4_dp;dense%n_components=2;sparse=dense
  a=dense%fit_transform(x);b=sparse%fit_transform(s)
  call require(maxval(abs(a-b))<1.e-8_dp,'CSR KernelPCA training coordinates')
  call require(.not.allocated(sparse%x_fit_).and.allocated(sparse%x_fit_csr_),'CSR KernelPCA owned sparse cache')
  call require(sparse%n_features_in_==3.and.sparse%n_samples_fit_==12,'CSR KernelPCA metadata')
  call require(all(s%data==before%data).and.all(s%indices==before%indices),'CSR KernelPCA input preservation')
  a=dense%transform(s);b=sparse%transform(x)
  call require(maxval(abs(a-b))<1.e-8_dp,'KernelPCA mixed query storage')
  call sparse%fit(x)
  call require(.not.sparse%sparse_fit_.and..not.allocated(sparse%x_fit_csr_),'KernelPCA dense refit clears CSR cache')
  sparse%eigen_solver='randomized';call sparse%fit(s)
  call require(.not.allocated(sparse%x_fit_),'KernelPCA sparse refit clears dense cache')
  if(arg=='--memory')then
    s=csr_from_arrays(8,50000,[(2*i+1,i=0,8)],[(i,i=1,16)],spread(0._dp,1,16))
    do i=1,8
      s%indices(2*i-1:2*i)=[50000,1];s%data(2*i-1:2*i)=[sin(real(i,dp)),cos(real(i,dp))]
    end do
    call sparse%fit(s)
    q%n_rows=1000;q%n_cols=50000;q%indptr=[(2*i+1,i=0,1000)]
    allocate(q%indices(2000),q%data(2000))
    do i=1,1000
      q%indices(2*i-1:2*i)=[50000,1];q%data(2*i-1:2*i)=[.3_dp,-.2_dp]
    end do
    a=sparse%transform(q)
    call require(all(shape(a)==[1000,2]),'wide CSR KernelPCA query shape')
    call require(.not.allocated(sparse%x_fit_),'wide CSR KernelPCA retains sparse design')
    print *,'KernelPCA CSR 1000 by 50000 memory: PASS'
  end if
  print *,'CSR KernelPCA fit, mixed queries, ownership and refits: PASS'
end program
