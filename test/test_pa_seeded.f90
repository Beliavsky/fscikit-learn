program test_pa_seeded
  use sklearn_common,only:dp,require
  use sklearn_linear_model,only:passive_aggressive_regressor,passive_aggressive_classifier
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(passive_aggressive_regressor)::r
  type(passive_aggressive_classifier)::c
  real(dp)::x(6,2),y(6),weights(3),expected(4),actual(4)
  integer::labels(6),nseed,i
  integer,allocatable::seed(:)
  character(32)::mode
  x(:,1)=[0._dp,-1._dp,2._dp,.5_dp,1._dp,-2._dp]
  x(:,2)=[0._dp,2._dp,1._dp,-.5_dp,1._dp,1._dp]
  y=2*x(:,1)-.5_dp*x(:,2)+.3_dp;labels=[-4,7,21,-4,7,21];weights=[.5_dp,2._dp,1.3_dp]
  call get_command_argument(1,mode)
  select case(trim(mode))
  case('reg_seed');r%random_state=-1;r%shuffle=.false.;call r%fit(x,y)
  case('class_seed');c%random_state=-1;c%shuffle=.false.;call c%fit(x,labels)
  case('reg_c');r%c=0;call r%fit(x,y)
  case('class_c');c%c=0;call c%fit(x,labels)
  case('epsilon');r%epsilon=-1;call r%fit(x,y)
  case('reg_loss');r%loss='squared_error';call r%fit(x,y)
  case('class_loss');c%loss='log_loss';call c%fit(x,labels)
  case('weight_nan');weights(1)=ieee_value(0._dp,ieee_quiet_nan);call c%fit(x,labels,weights)
  case('weight_inf');weights(1)=ieee_value(0._dp,ieee_positive_inf);call c%fit(x,labels,weights)
  case('weight_count');call c%fit(x,labels,weights(:2))
  case('weight_conflict');c%class_weight_balanced=.true.;call c%fit(x,labels,weights)
  end select
  call require(len_trim(mode)==0,'passive-aggressive guard unexpectedly accepted')
  call random_seed(size=nseed);allocate(seed(nseed));seed=[(19+17*i,i=1,nseed)]
  call random_seed(put=seed);call random_number(expected)
  call random_seed(put=seed)
  r%c=.3_dp;r%epsilon=.1_dp;r%max_iter=3;r%tol=-1;r%random_state=42;r%average=1
  call r%fit(x,y);call random_number(actual)
  call require(all(actual==expected),'PA regression does not modify the global RNG')
  call require(maxval(abs(r%coef_-[1.7233333333333333_dp,-.2966666666666667_dp]))<1.e-12_dp.and. &
    abs(r%intercept_-.2933333333333333_dp)<1.e-12_dp,'Source seeded averaged PA regression')
  call require(r%n_features_in_==2.and.r%t_==19,'PA regression fitted feature and epoch metadata')
  call random_seed(put=seed)
  c%c=.2_dp;c%max_iter=3;c%tol=-1;c%random_state=0
  call c%fit(x,labels,weights);call random_number(actual)
  call require(all(actual==expected),'PA classification does not modify the global RNG')
  call require(maxval(abs(c%coef_-reshape([.054_dp,-.036_dp,-.304_dp,-.822_dp,.972_dp,.068_dp],[3,2]))) &
    <1.e-12_dp.and.maxval(abs(c%intercept_-[-.172_dp,-.464_dp,-.236_dp]))<1.e-12_dp, &
    'Source independent OVR seeds and explicit class weights')
  call require(c%n_features_in_==2.and.c%t_==19,'PA classifier fitted feature and epoch metadata')
  call c%fit(x(:4,:),[-4,7,-4,7])
  call require(size(c%coef_,1)==1.and.size(c%classes_)==2,'PA refit resets multiclass heads')
  print *,'Seeded PA fitting, class weights, metadata and RNG isolation: PASS'
end program
