program test_sgd_validation
  use sklearn_common,only:dp,require
  use sklearn_sgd,only:sgd_regressor,sgd_classifier
  use sklearn_sgd_clone,only:clone_sgd_parameters
  use sklearn_parameter_space,only:parameter_grid,param
  use sklearn_search_adapters,only:set_search_params
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(sgd_regressor)::r,reference,copy
  type(sgd_classifier)::c,ccopy
  type(csr_matrix)::sx
  type(parameter_grid)::grid
  real(dp)::x(24,2),y(24),weights(24),changed(24)
  integer::labels(24),i
  character(32)::mode
  call get_command_argument(1,mode)
  do i=1,24
    x(i,:)=[real(i,dp)/24,sin(real(i,dp))]
    y(i)=.4_dp+x(i,1)-x(i,2);labels(i)=mod(i,3)
  end do
  weights=1;sx=csr_from_dense(x)
  r%early_stopping=.true.;r%validation_fraction=.25_dp;r%random_state=7
  r%max_iter=40;r%n_iter_no_change=2;r%tol=.1_dp
  c%early_stopping=.true.;c%validation_fraction=.25_dp;c%random_state=7
  c%max_iter=40;c%n_iter_no_change=2;c%tol=.1_dp;c%fit_intercept=.false.
  select case(trim(mode))
  case('rpartial');call r%partial_fit(x,y)
  case('rpartial_csr');call r%partial_fit(sx,y)
  case('cpartial');call c%partial_fit(x,labels,classes=[0,1,2])
  case('cpartial_csr');call c%partial_fit(sx,labels,classes=[0,1,2])
  case('fraction_zero');r%validation_fraction=0;call r%fit(x,y)
  case('fraction_one');r%validation_fraction=1;call r%fit(x,y)
  case('fraction_nan');r%validation_fraction=ieee_value(0._dp,ieee_quiet_nan);call r%fit(x,y)
  case('fraction_disabled');r%early_stopping=.false.;r%validation_fraction=0;call r%fit(x,y)
  case('empty_training');r%validation_fraction=.99_dp;call r%fit(x,y)
  case('singleton');labels=0;labels(1)=1;call c%fit(x,labels)
  case('small_validation');c%validation_fraction=.01_dp;call c%fit(x,labels)
  case('small_training');c%validation_fraction=.95_dp;call c%fit(x,labels)
  case('zero_validation')
    call r%fit(x,y);where(r%validation_mask_)weights=0
    call r%fit(x,y,weights)
  case('zero_validation_csr')
    call c%fit(sx,labels);where(c%validation_mask_)weights=0
    call c%fit(sx,labels,weights)
  end select
  call require(len_trim(mode)==0,'SGD validation guard unexpectedly accepted')
  call r%fit(x,y)
  call require(count(r%validation_mask_)==6,'SGD regression held-out row count')
  call require(r%t_==1+24*r%n_iter_,'SGD public t uses original sample count')
  call require(r%n_iter_<r%max_iter,'SGD validation patience stops regression')
  copy=clone_sgd_parameters(r)
  call require(copy%early_stopping.and.copy%validation_fraction==.25_dp,'SGD clone preserves validation controls')
  call require(.not.allocated(copy%coef_).and..not.allocated(copy%validation_mask_),'SGD clone discards fitted split')
  ! With one epoch the held-out targets can affect the score, never updates.
  r%max_iter=1;call r%fit(x,y);reference=r;changed=y
  where(r%validation_mask_)changed=changed+100
  call r%fit(x,changed)
  call require(maxval(abs(r%coef_-reference%coef_))==0.and.r%intercept_==reference%intercept_, &
    'held-out targets never enter SGD updates')
  call c%fit(x,labels);ccopy=c
  call require(count(c%validation_mask_)==6,'SGD stratified held-out row count')
  do i=0,2
    call require(count(c%validation_mask_.and.labels==i)==2,'SGD multiclass split is shared and stratified')
  end do
  call c%fit(sx,labels)
  call require(all(c%validation_mask_.eqv.ccopy%validation_mask_),'CSR shares deterministic validation split')
  call require(maxval(abs(c%coef_-ccopy%coef_))<1.e-14_dp,'dense/CSR no-intercept validation fitting')
  ccopy=clone_sgd_parameters(c)
  call require(ccopy%early_stopping.and..not.allocated(ccopy%coef_),'classifier clone preserves validation controls')
  r%early_stopping=.false.;call r%fit(x,y)
  call require(.not.any(r%validation_mask_),'disabled refit clears held-out rows')
  allocate(grid%axes(2))
  grid%axes(1)%name='early_stopping';grid%axes(1)%values=[param(.true.)]
  grid%axes(2)%name='validation_fraction';grid%axes(2)%values=[param(.35_dp)]
  call set_search_params(r,grid%at(1));call set_search_params(c,grid%at(1))
  call require(r%early_stopping.and.c%early_stopping.and.r%validation_fraction==.35_dp.and. &
    c%validation_fraction==.35_dp,'search accepts SGD validation controls')
  print *,'SGD validation splits, excluded rows, stopping, sparse fits and cloning: PASS'
end program
