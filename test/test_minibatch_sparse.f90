program test_minibatch_sparse
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_minibatch_nmf,only:mini_batch_nmf
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  type(mini_batch_nmf)::dense,sparse,ds,ss
  type(csr_matrix)::x,before,first,last
  real(dp)::a(12,7),w0(12,2),h0(2,7),betas(8)
  real(dp),allocatable::wd(:,:),ws(:,:)
  integer::i,j,b
  logical::halting,after
  character(32)::mode
  call get_command_argument(1,mode)
  if(mode=='memory'.or.mode=='memory_kl')then
    call memory_case(mode);stop
  end if
  do i=1,12
    do j=1,7
      a(i,j)=.2_dp+real(mod(i*j,9),dp)/7
    end do
    w0(i,:)=[.2_dp+i*.03_dp,.4_dp+i*.02_dp]
  end do
  do j=1,7
    h0(:,j)=[.3_dp+j*.01_dp,.6_dp+j*.02_dp]
  end do
  x=csr_from_dense(a);before=x;betas=[-.5_dp,0._dp,.5_dp,1._dp,1.5_dp,2._dp,2.5_dp,3._dp]
  dense%n_components=2;dense%init='custom';dense%max_iter=3;dense%batch_size=5;dense%tol=0
  dense%alpha_w=.01_dp;dense%alpha_h=.02_dp;dense%l1_ratio=.4_dp;dense%max_no_improvement=-1
  dense%fresh_restarts_max_iter=4;dense%transform_max_iter=5
  call ieee_get_halting_mode(ieee_invalid,halting)
  do b=1,size(betas)
    dense%beta_loss=betas(b);dense%fresh_restarts=mod(b,2)==0;sparse=dense
    wd=dense%fit_transform(a,w0,h0);ws=sparse%fit_transform(x,w0,h0)
    call require(maxval(abs(wd-ws))<1.e-8_dp,'MiniBatch CSR learned codes')
    call require(maxval(abs(dense%components_-sparse%components_))<1.e-8_dp,'MiniBatch CSR components')
    call require(maxval(abs(dense%components_numerator_-sparse%components_numerator_))<1.e-8_dp, &
      'MiniBatch CSR sufficient-statistic numerator')
    call require(maxval(abs(dense%components_denominator_-sparse%components_denominator_))<1.e-8_dp, &
      'MiniBatch CSR sufficient-statistic denominator')
    wd=dense%transform(a);ws=sparse%transform(x)
    call require(maxval(abs(wd-ws))<1.e-8_dp,'MiniBatch CSR inference')
    call require(dense%n_steps_==sparse%n_steps_.and.dense%n_iter_==sparse%n_iter_,'MiniBatch CSR counters')
    ds=dense;ss=sparse
    if(allocated(ds%components_))deallocate(ds%components_,ds%components_numerator_,ds%components_denominator_)
    if(allocated(ss%components_))deallocate(ss%components_,ss%components_numerator_,ss%components_denominator_)
    first=csr_from_dense(a(:4,:));last=csr_from_dense(a(5:,:))
    call ds%partial_fit(a(:4,:),w0(:4,:),h0);call ss%partial_fit(first,w0(:4,:),h0)
    call ds%partial_fit(a(5:,:));call ss%partial_fit(last)
    call require(maxval(abs(ds%components_-ss%components_))<1.e-8_dp,'MiniBatch CSR streaming updates')
  end do
  call ieee_get_halting_mode(ieee_invalid,after)
  call require(after.eqv.halting,'MiniBatch CSR restores IEEE halting mode')
  call require(all(x%indptr==before%indptr).and.all(x%indices==before%indices).and.all(x%data==before%data), &
    'MiniBatch CSR preserves input storage')
  print *,'MiniBatch CSR fits, restarts, sufficient statistics, streams and inference: PASS'
contains
  subroutine memory_case(mode)
    character(*),intent(in)::mode
    integer,parameter::n=384,p=200000,k=2
    type(mini_batch_nmf)::model
    type(csr_matrix)::design
    integer::row,ptr(n+1),columns(2*n)
    real(dp)::data(2*n)
    real(dp),allocatable::initial_w(:,:),initial_h(:,:),fit(:,:),query(:,:)
    ptr=[(2*row+1,row=0,n)];columns=[(row,row=1,2*n)];data=1
    design=csr_from_arrays(n,p,ptr,columns,data)
    allocate(initial_w(n,k),initial_h(k,p));initial_w=1;initial_h=.2_dp
    model%init='custom';model%n_components=k;model%max_iter=1;model%tol=0
    model%batch_size=64;model%transform_max_iter=1;model%fresh_restarts_max_iter=1;model%max_no_improvement=-1
    if(mode=='memory_kl')model%beta_loss=1
    fit=model%fit_transform(design,initial_w,initial_h);call model%partial_fit(design);query=model%transform(design)
    call require(all(ieee_is_finite(fit)).and.all(ieee_is_finite(query)),'large sparse MiniBatch finite factors')
    call require(sum(fit)>0.and.sum(query)>0,'large sparse MiniBatch positive factors')
    call require(model%n_features_in_==p.and.model%n_steps_==7,'large sparse MiniBatch dimensions/steps')
    call require(all(design%data==data),'large sparse MiniBatch input preservation')
    print *,'Large sparse MiniBatch memory case: PASS ',trim(mode)
  end subroutine
end program
