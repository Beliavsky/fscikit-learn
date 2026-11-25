program test_shrinkage
  use sklearn,only:dp,ledoit_wolf_shrinkage,ledoit_wolf
  use sklearn_common,only:require
  use,intrinsic::ieee_arithmetic,only:ieee_is_nan
  use,intrinsic::ieee_exceptions
  implicit none
  real(dp)::x(4,2),value,other
  real(dp),allocatable::large(:,:)
  logical::before(size(ieee_all)),after(size(ieee_all)),modes(size(ieee_all)),after_modes(size(ieee_all))
  type(ledoit_wolf)::model
  x(:,1)=[1._dp,-1._dp,0._dp,0._dp];x(:,2)=[0._dp,0._dp,1._dp,-1._dp]
  value=ledoit_wolf_shrinkage(x,assume_centered=.true.,block_size=1)
  call require(value==0,'isotropic covariance has zero denominator and zero shrinkage')
  x(:,2)=x(:,1)
  value=ledoit_wolf_shrinkage(x,assume_centered=.true.,block_size=1)
  other=ledoit_wolf_shrinkage(x,assume_centered=.true.,block_size=1000)
  call require(abs(value-.5_dp)<1.e-14_dp.and.abs(value-other)<1.e-14_dp,'rank-one analytic shrinkage')
  model%block_size=1;model%assume_centered=.true.;call model%fit(x)
  call require(abs(model%shrinkage_-value)<1.e-14_dp,'estimator uses blocked coefficient')
  call require(abs(model%covariance_(1,2)-.25_dp)<1.e-14_dp,'analytic shrunk off-diagonal')
  call ieee_set_flag(ieee_underflow,.true.)
  call ieee_get_flag(ieee_all,before);call ieee_get_halting_mode(ieee_all,modes)
  x=1.e160_dp
  value=ledoit_wolf_shrinkage(x,assume_centered=.true.,block_size=1)
  call ieee_get_flag(ieee_all,after);call ieee_get_halting_mode(ieee_all,after_modes)
  call require(ieee_is_nan(value),'overflow follows source quiet NaN result')
  call require(all(before.eqv.after).and.all(modes.eqv.after_modes),'all IEEE flags and modes preserved')
  value=ledoit_wolf_shrinkage(x(:,:1),block_size=1)
  call require(value==0,'single feature shortcut avoids unused overflow')
  allocate(large(1,8192));large=1
  value=ledoit_wolf_shrinkage(large,block_size=128)
  call require(value==0,'large one-sample centered shrinkage')
  print *,'blocked Ledoit-Wolf tests passed; avoided a 512 MiB cross-moment matrix'
end program
