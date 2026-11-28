program test_nmf_sparse
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_decomposition,only:nmf,non_negative_factorization
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  type(nmf)::dense,sparse
  type(csr_matrix)::x,before
  real(dp)::a(12,7),w0(12,2),h0(2,7),betas(5)
  real(dp),allocatable::wd(:,:),ws(:,:),h(:,:),z(:,:)
  integer::i,j,b,it
  character(32)::mode
  call get_command_argument(1,mode)
  if(mode=='memory_cd'.or.mode=='memory_mu')then
    call memory_case(mode);stop
  end if
  do i=1,12
    do j=1,7
      a(i,j)=real(mod(i*j,9),dp)/7
    end do
    w0(i,:)=[.2_dp+i*.03_dp,.4_dp+i*.02_dp]
  end do
  do j=1,7
    h0(:,j)=[.3_dp+j*.01_dp,.6_dp+j*.02_dp]
  end do
  x=csr_from_dense(a);before=x
  dense%solver='cd';dense%init='custom';dense%n_components=2;dense%max_iter=8;dense%tol=0
  sparse=dense;wd=dense%fit_transform(a,w0,h0);ws=sparse%fit_transform(x,w0,h0)
  call require(maxval(abs(wd-ws))<1.e-10_dp,'CSR coordinate descent W')
  call require(maxval(abs(dense%components_-sparse%components_))<1.e-10_dp,'CSR coordinate descent H')
  call require(maxval(abs(dense%transform(a)-sparse%transform(x)))<1.e-10_dp,'CSR coordinate transform')
  call non_negative_factorization(x,ws,h,it,n_components=2,w0=w0,h0=h0,init='custom', &
    solver='cd',max_iter=8,tol=0._dp)
  call require(it==8.and.maxval(abs(h-sparse%components_))<1.e-10_dp,'CSR functional factorization')
  betas=[.5_dp,1._dp,1.5_dp,2._dp,3._dp]
  do b=1,size(betas)
    dense%solver='mu';dense%beta_loss=betas(b);dense%alpha_w=.01_dp;dense%l1_ratio=.3_dp;sparse=dense
    wd=dense%fit_transform(a,w0,h0);ws=sparse%fit_transform(x,w0,h0)
    call require(maxval(abs(wd-ws))<1.e-9_dp,'CSR beta MU weights')
    call require(maxval(abs(dense%components_-sparse%components_))<1.e-9_dp,'CSR beta MU components')
    z=sparse%transform(x)
    call require(maxval(abs(z-dense%transform(a)))<1.e-9_dp,'CSR beta MU transform')
  end do
  call require(all(x%indptr==before%indptr).and.all(x%indices==before%indices).and.all(x%data==before%data), &
    'CSR factorization preserves caller storage')
  print *,'CSR NMF coordinate/MU fits, beta losses, transforms and functional API: PASS'
contains
  subroutine memory_case(mode)
    character(*),intent(in)::mode
    integer,parameter::n=128,p=500000,k=2
    type(nmf)::model
    type(csr_matrix)::design
    integer::row,ptr(n+1),columns(2*n)
    real(dp)::data(2*n)
    real(dp),allocatable::initial_w(:,:),initial_h(:,:),fit(:,:),query(:,:)
    ptr=[(2*row+1,row=0,n)];columns=[(row,row=1,2*n)];data=1
    design=csr_from_arrays(n,p,ptr,columns,data)
    allocate(initial_w(n,k),initial_h(k,p));initial_w=1;initial_h=.2_dp
    model%solver='cd';if(mode=='memory_mu')model%solver='mu'
    model%beta_loss=2;if(mode=='memory_mu')model%beta_loss=1
    model%init='custom';model%n_components=k;model%max_iter=1;model%tol=0
    fit=model%fit_transform(design,initial_w,initial_h);query=model%transform(design)
    call require(all(ieee_is_finite(fit)).and.all(ieee_is_finite(query)),'large CSR NMF finite factors')
    call require(sum(fit)>0.and.sum(query)>0,'large CSR NMF positive factors')
    call require(model%n_features_in_==p.and.model%n_iter_==1,'large CSR NMF metadata')
    call require(all(design%data==data),'large CSR NMF input preservation')
    print *,'Large CSR NMF memory case: PASS ',trim(mode)
  end subroutine
end program
