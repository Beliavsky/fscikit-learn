program test_gradient_boosting_warm_start
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:gradient_boosting_regressor,gradient_boosting_classifier
 use sklearn_classic_gradient_boosting,only:reset_boosting_fit
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan
 implicit none
 type(gradient_boosting_regressor)::reg,whole,copy
 type(gradient_boosting_classifier)::cls,all_cls,copy_cls
 real(dp)::x(19,3),y(19),w(19)
 integer::labels(19),i
 real(dp),allocatable::before(:),scores(:),raw(:,:)
 do i=1,19
  x(i,:)=[real(i,dp),real(mod(i,3),dp),real(mod(i,7),dp)]
  y(i)=real(i*i+mod(i,3),dp);w(i)=1+.5_dp*mod(i,4);labels(i)=mod(i,3)*17-9
 end do
 reg%n_estimators=2;reg%warm_start=.true.;reg%subsample=.65_dp;reg%random_state=7
 whole=reg;whole%n_estimators=5;whole%warm_start=.false.;call whole%fit(x,y,w)
 call reg%fit(x,y,w);before=reg%predict(x);scores=reg%train_score_
 call reg%fit(x,y,w)
 call require(reg%n_estimators_==3.and.size(reg%estimators_)==2,'equal-count warm fit preserves sklearn metadata quirk')
 call require(all(reg%predict(x)==before).and.all(reg%train_score_==scores),'equal-count call retains learned trees and scores')
 reg%n_estimators=5;call reg%fit(x,y,w)
 call require(maxval(abs(reg%predict(x)-whole%predict(x)))<1.e-11_dp,'warm regression RNG and stages match one-shot fitting')
 call require(maxval(abs(reg%oob_scores_-whole%oob_scores_))<1.e-10_dp,'warm regression OOB histories')
 copy=reg;call reset_boosting_fit(copy)
 call require(copy%warm_start.and..not.allocated(copy%estimators_),'clone reset preserves warm parameter and clears regression fit')
 call copy%fit(x,y,w)
 call require(maxval(abs(copy%predict(x)-whole%predict(x)))<1.e-11_dp,'reset clone refits independently from seed')
 cls%n_estimators=2;cls%warm_start=.true.;cls%subsample=.65_dp;cls%random_state=7
 all_cls=cls;all_cls%n_estimators=5;all_cls%warm_start=.false.;call all_cls%fit(x,labels,w)
 call cls%fit(x,labels,w);raw=cls%decision_function(x);call cls%fit(x,labels,w)
 call require(cls%n_estimators_==3.and.size(cls%estimators_,1)==2,'classifier equal-count metadata and actual stage count')
 call require(all(raw==cls%decision_function(x)).and.size(cls%apply(x),2)==2,'classifier queries use actual trees after equal-count fit')
 cls%n_estimators=5;call cls%fit(x,labels,w)
 call require(maxval(abs(cls%decision_function(x)-all_cls%decision_function(x)))<1.e-11_dp,'warm classifier matches one-shot fitting')
 copy_cls=cls;call reset_boosting_fit(copy_cls)
 call require(copy_cls%warm_start.and..not.allocated(copy_cls%classes_),'classifier clone keeps controls and clears learned labels')
 scores=reg%oob_scores_;reg%n_estimators=7;reg%subsample=1;call reg%fit(x,y,w)
 call require(all(reg%oob_scores_(:5)==scores).and.ieee_is_nan(reg%oob_score_),'warm full-data switch retains OOB arrays and NaN scalar')
 reg%warm_start=.false.;reg%n_estimators=3;call reg%fit(x,y,w)
 call require(size(reg%estimators_)==3.and..not.allocated(reg%oob_scores_),'cold refit discards warm history')
 print *,'Warm classic boosting: RNG continuity, equal counts, clone reset and OOB transitions PASS'
end program
