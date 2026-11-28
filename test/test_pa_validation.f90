program test_pa_validation
  use sklearn_common,only:dp,require
  use sklearn_linear_model,only:passive_aggressive_regressor,passive_aggressive_classifier
  use sklearn_sgd_clone,only:clone_pa_parameters
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(passive_aggressive_regressor)::r,other
  type(passive_aggressive_classifier)::c,copy
  type(csr_matrix)::s
  real(dp)::x(12,2),y(12),changed(12)
  real(dp),allocatable::data(:),large_y(:),prediction(:),scores(:,:)
  integer,allocatable::ptr(:),indices(:),large_labels(:)
  integer::labels(12),i
  character(40)::mode
  call get_command_argument(1,mode)
  r%early_stopping=.true.;c%early_stopping=.true.;r%random_state=42;c%random_state=42
  r%max_iter=3;c%max_iter=3;r%average=1;c%average=1;r%tol=-1;c%tol=-1
  r%validation_fraction=.25_dp;c%validation_fraction=.25_dp
  do i=1,12
    x(i,:)=[sin(real(i,dp)),cos(.3_dp*i)];y(i)=2*x(i,1)-x(i,2);labels(i)=mod(i,3)
  end do
  s=csr_from_dense(x)
  select case(trim(mode))
  case('reg_partial');call r%partial_fit(x,y)
  case('class_partial');call c%partial_fit(x,labels,classes=[0,1,2])
  case('reg_partial_csr');call r%partial_fit(s,y)
  case('class_partial_csr');call c%partial_fit(s,labels,classes=[0,1,2])
  case('reg_fraction_zero');r%validation_fraction=0;call r%fit(x,y)
  case('class_fraction_one');c%validation_fraction=1;call c%fit(x,labels)
  case('reg_fraction_nan');r%validation_fraction=ieee_value(0._dp,ieee_quiet_nan);call r%fit(x,y)
  case('class_fraction_disabled');c%early_stopping=.false.;c%validation_fraction=-.1_dp;call c%fit(x,labels)
  case('reg_empty_train');call r%fit(x(:1,:),y(:1))
  case('class_rare');call c%fit(x,[0,0,0,0,0,0,0,0,1,1,1,2])
  case('class_small_validation');c%validation_fraction=.1_dp;call c%fit(x,labels)
  case('class_small_train');c%validation_fraction=.9_dp;call c%fit(x,labels)
  case('memory')
    allocate(ptr(257),indices(512),data(512),large_y(256),large_labels(256))
    do i=1,256
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[i,500000-i];data(2*i-1:2*i)=[1._dp/(i+1),.5_dp]
      large_y(i)=sin(real(i,dp));large_labels(i)=mod(i,3)
    end do
    ptr(257)=513;s=csr_from_arrays(256,500000,ptr,indices,data)
    call r%fit(s,large_y);call c%fit(s,large_labels)
    prediction=r%predict(s);scores=c%decision_function(s)
    call require(count(r%validation_mask_)==64.and.count(c%validation_mask_)==64,'wide validation masks')
    call require(size(prediction)==256.and.all(shape(scores)==[256,3]),'wide validation queries')
    print *,'Large validation CSR passive-aggressive memory case: PASS';stop
  end select
  call require(len_trim(mode)==0,'PA validation guard unexpectedly accepted')
  call r%fit(x,y);call c%fit(s,labels)
  call require(count(r%validation_mask_)==3.and.count(c%validation_mask_)==3,'validation sizes')
  do i=0,2
    call require(count(c%validation_mask_.and.labels==i)==1,'stratified held-out classes')
  end do
  call require(r%t_==37.and.c%t_==37,'public clocks include held-out rows')
  changed=y;where(r%validation_mask_)changed=1.e6_dp
  other=clone_pa_parameters(r)
  call require(other%early_stopping.and.other%validation_fraction==.25_dp,'regression clone controls')
  call require(.not.allocated(other%validation_mask_),'regression clone drops held-out mask')
  call other%fit(x,changed)
  call require(all(other%coef_==r%coef_).and.other%intercept_==r%intercept_,'held-out targets never train')
  copy=clone_pa_parameters(c)
  call require(copy%early_stopping.and.copy%validation_fraction==.25_dp,'classifier clone controls')
  call require(.not.allocated(copy%validation_mask_),'classifier clone drops held-out mask')
  r%early_stopping=.false.;call r%fit(x,y)
  call require(.not.any(r%validation_mask_),'disabled validation uses all rows')
  print *,'PA validation masks, exclusions, sample clocks and clone controls: PASS'
end program
