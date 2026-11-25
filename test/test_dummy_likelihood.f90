program test_dummy_likelihood
  use sklearn_common,only:dp,require
  use sklearn_stateless_preprocessing,only:add_dummy_feature
  use sklearn_covariance_functions,only:log_likelihood
  use sklearn_sparse,only:csr_matrix,csr_from_arrays
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  real(dp)::x(2,2),identity(2,2),negative(2,2),score
  real(dp),allocatable::z(:,:)
  type(csr_matrix)::a,b
  logical::modes(size(ieee_all)),flags(size(ieee_all)),now(size(ieee_all)),expected(size(ieee_all))
  x=reshape([1._dp,2._dp,3._dp,4._dp],[2,2]);z=add_dummy_feature(x,-2._dp)
  call require(all(z(:,1)==-2).and.all(z(:,2:)==x),'dummy is first column and owns copied data')
  a=csr_from_arrays(2,2,[1,4,4],[2,1,2],[2._dp,0._dp,-2._dp]);b=add_dummy_feature(a,0._dp)
  call require(all(b%indptr==[1,4,5]).and.all(b%indices==[1,2,3,1]),'dummy CSR sorts, coalesces and keeps explicit zeros')
  call require(all(b%data==0),'cancelling duplicates and zero-valued dummy are stored')
  identity=0;identity(1,1)=1;identity(2,2)=1
  score=log_likelihood(identity,identity)
  call require(abs(score-(-1-log(2*acos(-1._dp))))<1.e-14_dp,'analytic normalized Gaussian likelihood')
  negative=-identity
  call require(abs(log_likelihood(identity,negative)-(1-log(2*acos(-1._dp))))<1.e-14_dp, &
    'positive determinant need not mean positive definite')
  negative(1,1)=1;score=log_likelihood(identity,negative)
  call require(.not.ieee_is_finite(score).and.score<0,'negative determinant gives negative infinity')
  call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
  call ieee_set_flag(ieee_underflow,.true.);call ieee_get_flag(ieee_all,expected)
  a=csr_from_arrays(1,1,[1,3],[1,1],[1e308_dp,1e308_dp]);b=add_dummy_feature(a)
  call require(.not.ieee_is_finite(b%data(2)),'duplicate overflow propagates without trapping')
  identity(1,1)=ieee_value(0._dp,ieee_quiet_nan);score=log_likelihood(identity,negative)
  call require(ieee_is_nan(score),'nonfinite trace propagates')
  call ieee_get_flag(ieee_all,now);call require(all(now.eqv.expected),'likelihood preserves caller IEEE flags')
  call require(log_likelihood(identity(:0,:0),identity(:0,:0))==0,'empty square likelihood is zero')
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  print *,'Dummy feature and general covariance likelihood tests passed'
end program
