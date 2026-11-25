program test_variance_scores
 use sklearn,only:dp,r2_score,explained_variance_score,r2_score_raw,explained_variance_score_raw
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 implicit none
 real(dp)::y(3),p(3),a(3,2),b(3,2),v
 real(dp),allocatable::raw(:)
 integer::n
 integer,allocatable::before(:),after(:)
 y=[1._dp,2._dp,3._dp];p=[1._dp,3._dp,3._dp]
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 call require(abs(r2_score(y,p)-.5_dp)<1.e-14_dp,'R2 ordinary')
 call require(abs(explained_variance_score(y,p)-2._dp/3)<1.e-14_dp,'explained variance ordinary')
 call require(explained_variance_score(y,y+2)==1,'explained variance ignores constant bias')
 call require(r2_score(y,y+2)<0,'R2 penalizes bias')
 a(:,1)=y;a(:,2)=2*y;b(:,1)=p;b(:,2)=2*p
 v=r2_score(a,b,multioutput=[1._dp,3._dp],raw_values=raw)
 call require(size(raw)==2.and.abs(v-.5_dp)<1.e-14_dp,'weighted outputs and owned raw result')
 call require(abs(r2_score(a,b,multioutput='variance_weighted')-.5_dp)<1.e-14_dp,'variance weighting')
 raw=r2_score_raw(a,b,sample_weight=[0._dp,1._dp,1._dp])
 call require(all(abs(raw+1)<1.e-14_dp),'weighted raw R2')
 call require(ieee_is_nan(r2_score(y(:1),p(:1),sample_weight=[0._dp])),'singleton R2 before zero weight check')
 call require(explained_variance_score(y(:1),p(:1))==1,'singleton explained variance')
 call require(r2_score([2._dp,2._dp],[1._dp,3._dp])==0,'finite constant R2')
 v=r2_score([2._dp,2._dp],[1._dp,3._dp],force_finite=.false.)
 call require(.not.ieee_is_finite(v).and.v<0,'constant R2 negative infinity')
 raw=explained_variance_score_raw(a,b)
 call require(all(abs(raw-2._dp/3)<1.e-14_dp),'raw explained variance')
 call require(r2_score(y,reshape(p,[3,1]))==r2_score(reshape(y,[3,1]),p),'mixed target ranks')
 call require(ieee_is_nan(r2_score([huge(1._dp),-huge(1._dp)],[0._dp,0._dp])),'overflow propagates quietly')
 call random_seed(get=after);call require(all(before==after),'variance metrics preserve RNG')
 print *, 'Weighted multioutput R2 and explained variance: PASS'
end program
