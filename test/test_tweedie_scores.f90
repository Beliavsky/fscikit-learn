program test_tweedie_scores
 use sklearn,only:dp,mean_tweedie_deviance,d2_tweedie_score,mean_poisson_deviance,mean_gamma_deviance
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 implicit none
 real(dp)::y(4),p(4),w(4),v,nan
 integer::n
 integer,allocatable::before(:),after(:)
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 y=[.5_dp,1._dp,2.5_dp,7._dp];p=[1._dp,1._dp,5._dp,3.5_dp];w=[1._dp,2._dp,0._dp,3._dp]
 call require(abs(d2_tweedie_score(y,p)-.2857142857142857_dp)<1.e-14_dp,'D2 normal source example')
 call require(d2_tweedie_score(y,y,power=2._dp)==1,'perfect Gamma D2')
 call require(abs(mean_tweedie_deviance(y,p,w,power=1._dp)-mean_poisson_deviance(y,p,w))<1.e-14_dp, &
  'weighted Poisson/Tweedie identity')
 call require(abs(mean_tweedie_deviance(y,p,w,power=2._dp)-mean_gamma_deviance(y,p,w))<1.e-14_dp, &
  'weighted Gamma/Tweedie identity')
 call require(mean_tweedie_deviance(reshape(y,[4,1]),p)==mean_tweedie_deviance(y,reshape(p,[4,1])), &
  'mixed column/vector ranks')
 call require(mean_tweedie_deviance(y,p,sample_weight=[-1._dp,2._dp,3._dp,4._dp])>=0,'signed sample weights')
 nan=ieee_value(0._dp,ieee_quiet_nan)
 call require(ieee_is_nan(mean_tweedie_deviance(y,p,sample_weight=[nan,1._dp,1._dp,1._dp])), &
  'nonfinite weights propagate')
 call require(ieee_is_nan(d2_tweedie_score([0._dp],[-1._dp],sample_weight=[1._dp,2._dp],power=2._dp)), &
  'singleton D2 precedes weight-length/domain checks')
 call require(ieee_is_nan(d2_tweedie_score([0._dp,0._dp],[1._dp,1._dp],power=1._dp)), &
  'zero Poisson null prediction retains xlogy NaN')
 v=mean_tweedie_deviance([-2._dp,0._dp,1._dp],[1._dp,1._dp,1._dp],power=-1.5_dp)
 call require(ieee_is_finite(v),'negative power permits negative truth')
 v=mean_tweedie_deviance([huge(1._dp)],[-huge(1._dp)])
 call require(.not.ieee_is_finite(v),'quiet deviance overflow')
 call random_seed(get=after);call require(all(before==after),'Tweedie metrics preserve RNG')
 print *, 'General weighted Tweedie deviance and D2 column scores: PASS'
end program
