program test_pa_csr
  use sklearn_common,only:dp,require
  use sklearn_linear_model,only:passive_aggressive_regressor,passive_aggressive_classifier
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use,intrinsic::ieee_exceptions
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(passive_aggressive_regressor)::r,d
  type(passive_aggressive_classifier)::c
  type(csr_matrix)::x,q,original
  real(dp)::dense(6,2),y(6),nan,overflow_input(6,4)
  logical::modes(size(ieee_all)),after_modes(size(ieee_all)),overflow_flag
  real(dp),allocatable::prediction(:),target(:),data(:),scores(:,:)
  integer,allocatable::labels(:),ptr(:),indices(:)
  integer::i
  character(32)::mode
  call get_command_argument(1,mode)
  if(trim(mode)=='memory')then
    allocate(ptr(513),indices(1024),data(1024),target(512),labels(512))
    do i=1,512
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[1+mod(i,200),500000-mod(i,300)]
      data(2*i-1:2*i)=[1._dp/(i+1),.5_dp];target(i)=sin(real(i,dp));labels(i)=mod(i,3)
    end do
    ptr(513)=1025;x=csr_from_arrays(512,500000,ptr,indices,data)
    r%max_iter=2;r%tol=-1;r%average=1;c%max_iter=2;c%tol=-1;c%average=1
    call r%fit(x,target);prediction=r%predict(x)
    call c%fit(x,labels);scores=c%decision_function(x)
    call require(size(prediction)==512.and.all(shape(scores)==[512,3]),'large CSR PA fit/query shapes')
    call require(r%n_features_in_==500000.and.c%n_features_in_==500000,'large CSR PA feature counts')
    print *,'Large CSR passive-aggressive memory case: PASS';stop
  end if
  dense(:,1)=[0._dp,-1._dp,2._dp,.5_dp,1._dp,-2._dp]
  dense(:,2)=[0._dp,2._dp,1._dp,-.5_dp,1._dp,1._dp]
  y=2*dense(:,1)-.5_dp*dense(:,2)+.3_dp;x=csr_from_dense(dense);q=x;original=x
  nan=ieee_value(0._dp,ieee_quiet_nan)
  select case(trim(mode))
  case('reg_unfitted');prediction=r%predict(x)
  case('class_unfitted');scores=c%decision_function(x)
  case('unallocated');deallocate(x%indptr);call r%fit(x,y)
  case('empty');x=csr_from_arrays(0,2,[1],[integer::],[real(dp)::]);call r%fit(x,[real(dp)::])
  case('zero_width');x=csr_from_arrays(6,0,[1,1,1,1,1,1,1],[integer::],[real(dp)::]);call r%fit(x,y)
  case('nan');x%data(1)=nan;call r%fit(x,y)
  case('inf');x%data(1)=ieee_value(0._dp,ieee_positive_inf);call r%fit(x,y)
  case('pointer');x%indptr(1)=0;call r%fit(x,y)
  case('column');x%indices(1)=3;call r%fit(x,y)
  case('target_count');call r%fit(x,y(:5))
  case('target_nan');y(1)=nan;call r%fit(x,y)
  case('class_count');call c%fit(x,[0,1])
  case('reg_width');call r%fit(x,y);q%n_cols=3;prediction=r%predict(q)
  case('class_width');call c%fit(x,[0,1,2,0,1,2]);q%n_cols=3;scores=c%decision_function(q)
  case('query_nan');call r%fit(x,y);q%data(1)=nan;prediction=r%predict(q)
  case('fit_weight');call r%fit(x,y,sample_weight=[1._dp,1._dp,1._dp,1._dp,1._dp,1._dp])
  end select
  call require(len_trim(mode)==0,'CSR PA guard unexpectedly accepted')
  r%c=.2_dp;r%epsilon=.5_dp;r%max_iter=1;r%tol=-1;r%shuffle=.false.
  x=csr_from_arrays(1,2,[1,3],[1,2],[2._dp,1._dp])
  call r%fit(x,[3._dp])
  call require(maxval(abs(r%coef_-[.4_dp,.2_dp]))<1.e-14_dp.and.abs(r%intercept_-.002_dp)<1.e-14_dp, &
    'CSR PA-I closed-form update and sparse intercept decay')
  r%loss='squared_epsilon_insensitive';call r%fit(x,[3._dp])
  call require(maxval(abs(r%coef_-[2._dp/3,1._dp/3]))<1.e-14_dp.and.abs(r%intercept_-1._dp/300)<1.e-14_dp, &
    'CSR PA-II closed-form update')
  x=csr_from_arrays(1,1,[1,3],[1,1],[1._dp,1._dp])
  r%loss='epsilon_insensitive';r%c=10;r%epsilon=0;r%fit_intercept=.false.
  call r%fit(x,[3._dp])
  call require(abs(r%coef_(1)-3)<1.e-14_dp,'CSR duplicate norms retain individual stored nodes')
  x=original;r%fit_intercept=.false.;d%fit_intercept=.false.;r%average=1;d%average=1
  r%max_iter=5;d%max_iter=5;r%tol=-1;d%tol=-1;r%shuffle=.true.;d%shuffle=.true.
  d%c=r%c;d%epsilon=r%epsilon
  call r%fit(x,y);call d%fit(dense,y)
  call require(maxval(abs(r%coef_-d%coef_))<1.e-12_dp,'dense/CSR no-bias PA averaged updates')
  call require(all(x%data==original%data).and.all(x%indices==original%indices).and. &
    all(x%indptr==original%indptr),'CSR PA preserves caller storage')
  call c%fit(x,[-4,7,21,-4,7,21]);prediction=real(c%predict(x),dp)
  call require(all(prediction==c%predict(dense)),'CSR PA classifier dense and sparse queries')
  call r%fit(dense,y);prediction=r%predict(x)
  call require(maxval(abs(prediction-r%predict(dense)))<1.e-12_dp,'CSR query after dense PA fit')
  call r%fit(x,y);call r%fit(dense,y)
  call require(r%n_features_in_==2,'PA dense/CSR refit metadata')
  overflow_input=1.e160_dp;r%max_iter=1;r%fit_intercept=.true.;r%average=1
  x=csr_from_dense(overflow_input)
  call ieee_get_halting_mode(ieee_all,modes);call ieee_set_flag(ieee_overflow,.false.)
  call r%fit(x,y);call ieee_get_halting_mode(ieee_all,after_modes);call ieee_get_flag(ieee_overflow,overflow_flag)
  call require(all(r%coef_==0).and.r%intercept_==0,'Source infinite CSR norm takes a zero PA step')
  call require(all(modes.eqv.after_modes).and..not.overflow_flag,'PA norm restores IEEE modes/overflow flag')
  call r%fit(overflow_input,y)
  call require(all(r%coef_==0).and.r%intercept_==0,'Source infinite dense norm takes a zero PA step')
  print *,'CSR passive-aggressive fitting, nodes, averaging, queries and refits: PASS'
end program
