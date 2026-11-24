program test_probability_metrics
  use sklearn_metrics, only: log_loss,d2_log_loss_score
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_nan,ieee_is_negative
  implicit none
  integer :: y(4)=[0,1,1,0],indicator(2,3)
  real(dp) :: probability(4)=[.1_dp,.8_dp,.7_dp,.2_dp],p(4,2),v,baseline,pm(2,3)
  p(:,1)=1-probability;p(:,2)=probability
  baseline=-sum(log([.9_dp,.8_dp,.7_dp,.8_dp]))/4
  call require(abs(log_loss(y,probability)-baseline)<1.e-14_dp,'Binary probability vector cross-entropy')
  call require(abs(log_loss(y,p)-baseline)<1.e-14_dp,'Binary probability matrix cross-entropy')
  call require(abs(log_loss(y,p,normalize=.false.)-4*baseline)<1.e-14_dp,'Summed cross-entropy')
  call require(abs(log_loss(y,p,sample_weight=[1._dp,2._dp,0._dp,1._dp])- &
    (-log(.9_dp)-3*log(.8_dp))/4)<1.e-14_dp,'Weighted cross-entropy')
  p=.8_dp
  call require(abs(log_loss(y,p)+log(.8_dp))<1.e-14_dp,'Rows are clipped without renormalization')
  v=log_loss(reshape([0,0,0,0],[2,2]),p(:2,:))
  call require(v==0.and..not.ieee_is_negative(v),'Empty indicator losses have positive-zero reduction identity')
  call require(abs(log_loss([1,0],reshape([.8_dp,.2_dp,.2_dp,.8_dp],[2,2]),labels=[1,0])- &
    (-log(.2_dp)))<1.e-14_dp,'Explicit classes are sorted rather than used in argument order')
  call require(abs(d2_log_loss_score(y,probability)-(1-baseline/log(2._dp)))<1.e-14_dp,'Fraction of log loss explained')
  call require(abs(d2_log_loss_score(y,[.5_dp,.5_dp,.5_dp,.5_dp]))<1.e-14_dp,'Null model has zero D2')
  call require(ieee_is_nan(d2_log_loss_score([9],[.3_dp])),'Single-sample D2 is undefined before class checks')
  indicator(1,:)=[1,0,1];indicator(2,:)=[0,1,0]
  pm(1,:)=[.7_dp,.2_dp,.8_dp];pm(2,:)=[.1_dp,.9_dp,.3_dp]
  call require(abs(log_loss(indicator,pm)-(-log(.7_dp)-log(.8_dp)-log(.9_dp))/2)<1.e-14_dp, &
    'Dense multilabel indicators retain all relevant-class log terms')
  indicator(1,:)=[3,4,3];indicator(2,:)=[4,3,4]
  call require(abs(log_loss(indicator,pm)-(-3*log(.7_dp)-4*log(.2_dp)-3*log(.8_dp)- &
    4*log(.1_dp)-3*log(.9_dp)-4*log(.3_dp))/2)<1.e-13_dp,'Source nonbinary indicator coefficient behavior')
  v=log_loss(y,probability,sample_weight=[1._dp,ieee_value(0._dp,ieee_quiet_nan),1._dp,1._dp])
  call require(ieee_is_nan(v),'NaN log-loss weights compose quietly')
  print *, 'Probability log loss, weighted sums, indicator targets and D2 scoring: PASS'
end program
