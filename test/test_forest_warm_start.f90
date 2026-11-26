program test_forest_warm_start
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:random_forest_regressor,extra_trees_regressor,random_forest_classifier,extra_trees_classifier
 use sklearn_tree,only:tree_probability
 implicit none
 type(random_forest_regressor)::reg,cold
 type(extra_trees_regressor)::extra,extra_cold
 type(random_forest_classifier)::cls,cls_cold
 type(extra_trees_classifier)::ec,ec_cold
 type(tree_probability),allocatable::prob(:),expected(:)
 real(dp)::x(20,2),query(20,2),y(20,2),w(20),cached_score
 real(dp),allocatable::before(:,:),prediction(:,:),oob(:,:)
 integer::labels(20,2),i,k
 integer,allocatable::bags(:,:),counts(:)
 do i=1,20
  x(i,:)=[real(i,dp),real(mod(i,3),dp)]
  y(i,:)=[real(i*i,dp),real(21-i,dp)];labels(i,:)=[mod(i,2),merge(0,1,i<=10)]
 end do
 query=x+.17_dp;w=1;w(2)=.5_dp;w(4)=2
 reg%warm_start=.true.;reg%random_state=7;reg%oob_score=.true.;reg%n_estimators=2
 call reg%fit(x,y,w);reg%n_estimators=5;call reg%fit(x,y,w)
 reg%n_estimators=8;call reg%fit(x,y,w)
 cold%random_state=7;cold%oob_score=.true.;cold%n_estimators=8;call cold%fit(x,y,w)
 call require(all(reg%apply(query)==cold%apply(query)),'warm regression seeds and component geometry match one-shot fit')
 prediction=reg%predict_multi(query);before=cold%predict_multi(query)
 call require(maxval(abs(prediction-before))<1.e-12_dp,'warm regression predictions match one-shot fit')
 bags=reg%estimators_samples();call require(all(bags==cold%estimators_samples()),'warm regression bootstrap sequence')
 call require(all(reg%oob_counts_==cold%oob_counts_),'warm regression OOB coverage uses entire ensemble')
 call require(maxval(abs(reg%oob_prediction_multi_-cold%oob_prediction_multi_))<1.e-12_dp,'warm regression OOB averages')
 cached_score=reg%oob_score_;oob=reg%oob_prediction_multi_;counts=reg%oob_counts_
 call reg%fit(x,2*y,2*w);before=reg%predict_multi(query)
 call require(all(before==prediction),'unchanged tree count does not refit old regression trees')
 call require(reg%oob_score_==cached_score.and.all(reg%oob_prediction_multi_==oob),'no-growth fit retains cached OOB')
 reg%oob_score=.false.;reg%n_estimators=9;call reg%fit(x,2*y,w)
 call require(reg%oob_score_==cached_score.and.all(reg%oob_counts_==counts),'disabled OOB retains cache while growing')
 reg%warm_start=.false.;reg%n_estimators=3;call reg%fit(x,y,w)
 cold%oob_score=.false.;cold%n_estimators=3;call cold%fit(x,y,w)
 before=reg%predict_multi(query);prediction=cold%predict_multi(query)
 call require(all(before==prediction),'disabling warm start fits a fresh regression ensemble')
 call require(reg%oob_score_==cached_score,'cold fit without OOB preserves last computed OOB score')
 extra%warm_start=.true.;extra%random_state=42;extra%n_estimators=2;call extra%fit(x,y)
 extra%n_estimators=6;call extra%fit(x,y)
 extra_cold%random_state=42;extra_cold%n_estimators=6;call extra_cold%fit(x,y)
 call require(all(extra%apply(query)==extra_cold%apply(query)),'warm ExtraTrees regression geometry')
 cls%warm_start=.true.;cls%random_state=7;cls%oob_score=.true.;cls%class_weight='balanced_subsample'
 cls%n_estimators=2;call cls%fit(x,labels,w);cls%n_estimators=6;call cls%fit(x,labels,w)
 cls_cold%random_state=7;cls_cold%oob_score=.true.;cls_cold%class_weight='balanced_subsample'
 cls_cold%n_estimators=6;call cls_cold%fit(x,labels,w)
 prob=cls%predict_proba_multi(query);expected=cls_cold%predict_proba_multi(query)
 do k=1,2
  call require(all(prob(k)%values==expected(k)%values),'warm classification probabilities match one-shot fit')
  call require(all(cls%oob_decision_function_multi_(k)%values==cls_cold%oob_decision_function_multi_(k)%values), &
   'warm classification OOB probabilities use entire ensemble')
 end do
 cls%criterion='entropy';cls%max_depth=1;cls%n_estimators=7;call cls%fit(x,labels,w)
 call require(cls%estimators_(1)%criterion=='gini'.and.cls%estimators_(1)%max_depth==100,'old tree settings are retained')
 call require(cls%estimators_(7)%criterion=='entropy'.and.cls%estimators_(7)%max_depth==1,'new trees use updated controls')
 ec%warm_start=.true.;ec%random_state=42;ec%n_estimators=2;call ec%fit(x,labels)
 ec%n_estimators=6;call ec%fit(x,labels)
 ec_cold%random_state=42;ec_cold%n_estimators=6;call ec_cold%fit(x,labels)
 call require(all(ec%apply(query)==ec_cold%apply(query)),'warm ExtraTrees classification geometry')
 print *,'Four warm-start forests, one-shot equivalence, unchanged prefixes, OOB caching and reset: PASS'
end program
