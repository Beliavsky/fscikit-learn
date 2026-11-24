program test_power_edges
 use sklearn,only:dp,power_transformer,power_transform
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 type(power_transformer)::model
 real(dp)::x(5,1),q(3,1),saved(5,1)
 real(dp),allocatable::z(:,:),a(:,:),b(:,:),mean(:),scale(:)
 logical::h0(3),h1(3),f0(3),f1(3)
 type(ieee_flag_type)::flags(3)
 integer::i
 flags=[ieee_invalid,ieee_divide_by_zero,ieee_overflow]
 x(:,1)=[.1_dp,.5_dp,1._dp,2._dp,4._dp];saved=x
 z=power_transform(x);a=model%fit_transform(x)
 call require(maxval(abs(z-a))==0.and.all(x==saved),'free power transform and input ownership')
 mean=model%mean_;scale=model%scale_
 model%standardize=.false.;call model%fit(x)
 call require(all(model%mean_==mean).and.all(model%scale_==scale),'nonstandard refit retains cached scaler')
 model%standardize=.true.;a=model%transform(x)
 model%standardize=.false.;b=model%transform(x)
 call require(maxval(abs(a(:,1)-(b(:,1)-mean(1))/scale(1)))<1.e-12_dp,'live retained scaler')
 model%lambdas_=-1;q(:,1)=[1._dp,2._dp,huge(1._dp)]
 do i=1,3
  call ieee_get_halting_mode(flags(i),h0(i));call ieee_get_flag(flags(i),f0(i))
 end do
 a=model%inverse_transform(q)
 do i=1,3
  call ieee_get_halting_mode(flags(i),h1(i));call ieee_get_flag(flags(i),f1(i))
 end do
 call require(all(h0.eqv.h1).and.all(f0.eqv.f1),'power inverse preserves IEEE policy')
 call require(.not.ieee_is_finite(a(1,1)).and.a(2,1)==-2._dp,'Yeo inverse boundary and integer reciprocal')
 model%method='box-cox';a=model%inverse_transform(q)
 call require(.not.ieee_is_finite(a(1,1)).and.ieee_is_nan(a(2,1)),'Box-Cox inverse domain yields NaN/Inf')
 call ieee_set_halting_mode(ieee_invalid,.false.);call ieee_set_flag(ieee_invalid,.true.)
 a=model%inverse_transform(q)
 call ieee_get_flag(ieee_invalid,f1(1));call ieee_get_halting_mode(ieee_invalid,h1(1))
 call require(f1(1).and..not.h1(1),'power preserves pre-existing raised flag')
 call ieee_set_flag(ieee_invalid,f0(1));call ieee_set_halting_mode(ieee_invalid,h0(1))
 model%method='yeo-johnson';model%lambdas_=ieee_value(0._dp,ieee_quiet_nan)
 q(:,1)=[-1._dp,0._dp,1._dp];a=model%transform(q)
 call require(abs(a(1,1)+log(2._dp))<1.e-14_dp.and.all(ieee_is_nan(a(2:,1))), &
  'source NaN power negative logarithmic branch')
 model%lambdas_=ieee_value(0._dp,ieee_positive_inf);q=ieee_value(0._dp,ieee_quiet_nan)
 a=model%inverse_transform(q);call require(all(a==0),'source infinite power NaN inverse')
 ! fit_transform skips a near-zero constant Yeo column, as in sklearn.
 model%method='yeo-johnson';x=1.e-20_dp
 a=model%fit_transform(x);b=model%transform(x)
 call require(all(a==x).and.all(b==0),'source constant fit_transform shortcut')
 print *, 'Power edge tests PASS'
end program
