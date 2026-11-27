program test_gradient_boosting_subsample
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:gradient_boosting_regressor,gradient_boosting_classifier
 use,intrinsic::ieee_arithmetic,only:ieee_is_finite
 implicit none
 type(gradient_boosting_regressor)::reg
 type(gradient_boosting_classifier)::cls
 real(dp)::x(12,2),y(12),w(12)
 real(dp),allocatable::prediction(:),again(:),scores(:),stages(:,:),raw(:,:),raw_stages(:,:,:)
 integer::labels(12),i,k
 do i=1,12
  x(i,:)=[real(i,dp),real(mod(i,5),dp)];y(i)=real(i*i,dp);labels(i)=mod(i,3)*17-9
 end do
 w=1;reg%n_estimators=4;reg%subsample=.5_dp;reg%random_state=7;reg%max_depth=2
 call reg%fit(x,y,w);prediction=reg%predict(x);scores=reg%oob_scores_;stages=reg%staged_predict(x)
 call require(all(ieee_is_finite(scores)).and.all(ieee_is_finite(reg%oob_improvement_)),'finite regression OOB histories')
 call require(all(stages(:,4)==prediction).and.reg%oob_score_==scores(4),'final regression stage and OOB score')
 do i=1,4
  call require(reg%estimators_(i)%nodes_(1)%n_samples==6,'regression samples exactly floor(subsample*n) rows')
 end do
 call reg%fit(x,y,w);again=reg%predict(x)
 call require(all(prediction==again).and.all(scores==reg%oob_scores_),'seeded stochastic regression refit')
 cls%n_estimators=4;cls%subsample=.5_dp;cls%random_state=7;cls%max_depth=2
 call cls%fit(x,labels,w);raw=cls%decision_function(x);raw_stages=cls%staged_decision_function(x)
 call require(all(raw_stages(:,:,4)==raw),'stochastic classification stages reach final decision')
 call require(all(ieee_is_finite(cls%oob_scores_)).and.cls%oob_score_==cls%oob_scores_(4),'classification OOB history')
 do i=1,4
  do k=1,3
   call require(cls%estimators_(i,k)%nodes_(1)%n_samples==6,'all class channels share the stage row mask')
  end do
 end do
 reg%subsample=.001_dp;call reg%fit(x,y)
 do i=1,4
  call require(reg%estimators_(i)%nodes_(1)%n_samples==1,'tiny fractions retain at least one in-bag row')
 end do
 reg%subsample=1;call reg%fit(x,y)
 cls%subsample=1;call cls%fit(x,labels)
 call require(.not.allocated(reg%oob_scores_).and..not.allocated(cls%oob_scores_),'full-data refits clear OOB arrays')
 call require(.not.allocated(reg%oob_improvement_).and..not.allocated(cls%oob_improvement_),'full-data refits clear improvements')
 print *,'Stochastic classic boosting: sampled stages, OOB histories and refitting PASS'
end program
