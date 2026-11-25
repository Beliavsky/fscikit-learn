program test_quantile_losses
 use sklearn,only:dp,mean_pinball_loss,mean_pinball_loss_raw,d2_pinball_score,d2_pinball_score_raw, &
  d2_absolute_error_score,d2_absolute_error_score_raw
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 implicit none
 real(dp)::y(3),p(3),a(3,2),b(3,2),v,nan
 real(dp),allocatable::raw(:),empty_weights(:)
 integer::n
 integer,allocatable::before(:),after(:)
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 y=[1._dp,2._dp,3._dp];p=[1._dp,3._dp,3._dp]
 call require(abs(mean_pinball_loss(y,p)-1._dp/6)<1.e-14_dp,'median pinball')
 call require(d2_pinball_score(y,p)==.5_dp,'pinball D2')
 call require(d2_absolute_error_score(y,p)==.5_dp,'absolute D2')
 a(:,1)=y;a(:,2)=y;b(:,1)=p;b(:,2)=y
 v=mean_pinball_loss(a,b,output_weights=[1._dp,3._dp],raw_values=raw)
 call require(abs(v-1._dp/24)<1.e-14_dp.and.raw(2)==0,'legacy output weighting and owned raw')
 call require(v==mean_pinball_loss(a,b,multioutput=[1._dp,3._dp]),'modern output weight keyword')
 call require(mean_pinball_loss(a,b,empty_weights,output_weights=empty_weights)==mean_pinball_loss(a,b), &
  'legacy optional unallocated weights')
 raw=mean_pinball_loss_raw(a,b);call require(size(raw)==2,'raw loss helper')
 raw=d2_absolute_error_score_raw(a,b);call require(raw(2)==1,'raw absolute score')
 v=d2_pinball_score(a(:1,:),b(:1,:),output_weights=[1._dp,-1._dp],raw_values=raw)
 call require(ieee_is_nan(v).and.size(raw)==2.and.all(ieee_is_nan(raw)),'legacy singleton raw width')
 raw=d2_pinball_score_raw(a(:1,:),b(:1,:));call require(size(raw)==1.and.ieee_is_nan(raw(1)), &
  'source singleton scalar adapted by raw helper')
 call require(mean_pinball_loss(y,reshape(p,[3,1]))==mean_pinball_loss(reshape(y,[3,1]),p),'mixed rank loss')
 nan=ieee_value(0._dp,ieee_quiet_nan)
 call require(ieee_is_nan(mean_pinball_loss(y,p,sample_weight=[nan,1._dp,1._dp])),'loss nonfinite weights')
 call require(ieee_is_nan(d2_pinball_score([1._dp],[2._dp],sample_weight=[nan])),'singleton skips CDF weight checks')
 v=d2_pinball_score([huge(1._dp),huge(1._dp)],[-huge(1._dp),-huge(1._dp)])
 call require(v==0,'NaN numerator with zero denominator follows source zero score')
 call random_seed(get=after);call require(all(before==after),'quantile metrics preserve RNG')
 print *, 'Weighted pinball and D2 quantile/absolute scores: PASS'
end program
