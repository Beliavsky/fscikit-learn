program test_covariance_functions
  use sklearn_common,only:dp,require
  use sklearn_covariance,only:empirical_covariance,shrunk_covariance,ledoit_wolf,oas,covariance_estimate
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  type(empirical_covariance)::emp
  type(shrunk_covariance)::shrunk
  type(ledoit_wolf)::lw
  type(oas)::oracle
  type(covariance_estimate)::estimate,copy
  real(dp)::x(3,2),expected(2,2),a(2,2),stack(3,2,2)
  real(dp),allocatable::cov(:,:),batch(:,:,:)
  logical::flags(size(ieee_all)),modes(size(ieee_all)),after(size(ieee_all)),after_modes(size(ieee_all))
  x(:,1)=[1._dp,0._dp,1._dp];x(:,2)=[0._dp,1._dp,1._dp]
  expected(1,:)=[2._dp/9,-1._dp/9];expected(2,:)=[-1._dp/9,2._dp/9]
  cov=empirical_covariance(x)
  call require(maxval(abs(cov-expected))<1.e-15_dp,'functional biased empirical covariance')
  cov=empirical_covariance([1._dp,2._dp,3._dp])
  call require(all(shape(cov)==[3,3]).and.all(cov==0),'vector means one sample with multiple features')
  cov=empirical_covariance([1._dp,2._dp],assume_centered=.true.)
  call require(all(cov(1,:)==[1._dp,2._dp]).and.all(cov(2,:)==[2._dp,4._dp]),'uncentered vector outer product')
  emp=empirical_covariance(assume_centered=.true.);call emp%fit(x)
  cov=empirical_covariance(x,assume_centered=.true.)
  call require(maxval(abs(cov-emp%covariance_))<1.e-15_dp,'type constructor remains available')
  shrunk=shrunk_covariance(shrinkage=.25_dp);call shrunk%fit(x)
  cov=shrunk_covariance(expected,.25_dp)
  call require(maxval(abs(cov-shrunk%covariance_))<1.e-15_dp,'shrinkage function and constructor coexist')
  lw=ledoit_wolf(block_size=2);call lw%fit(x);estimate=ledoit_wolf(x,block_size=2)
  call require(abs(estimate%shrinkage-lw%shrinkage_)<1.e-14_dp,'Ledoit-Wolf structured tuple coefficient')
  call require(maxval(abs(estimate%covariance-lw%covariance_))<1.e-14_dp,'Ledoit-Wolf structured tuple covariance')
  oracle=oas();call oracle%fit(x);estimate=oas(x);copy=estimate
  call require(maxval(abs(estimate%covariance-oracle%covariance_))<1.e-14_dp,'OAS function and constructor coexist')
  estimate%covariance=0
  call require(maxval(abs(copy%covariance-oracle%covariance_))<1.e-14_dp,'owned structured covariance result')
  a(1,:)=[2._dp,3._dp];a(2,:)=[7._dp,-4._dp]
  stack(1,:,:)=a;stack(2,:,:)=2*a;stack(3,:,:)=0
  batch=shrunk_covariance(stack,.5_dp)
  call require(all(batch(1,1,:)==[.5_dp,1.5_dp]).and.all(batch(1,2,:)==[3.5_dp,-2.5_dp]),'asymmetric batched shrinkage')
  call require(all(batch(2,:,:)==2*batch(1,:,:)).and.all(batch(3,:,:)==0),'independent leading batch axes')
  call ieee_set_flag(ieee_all,.false.);call ieee_set_flag(ieee_underflow,.true.)
  call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
  x(1,1)=ieee_value(0._dp,ieee_quiet_nan);cov=empirical_covariance(x)
  call ieee_get_flag(ieee_all,after);call ieee_get_halting_mode(ieee_all,after_modes)
  call require(all(flags.eqv.after).and.all(modes.eqv.after_modes),'functional covariance restores IEEE state')
  call require(all(ieee_is_nan(cov(1,:))).and.all(ieee_is_nan(cov(:,1))),'empirical nonfinite input propagates')
  print *,'Covariance functions and estimator constructors passed'
end program
