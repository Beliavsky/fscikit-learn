program test_gradient_boosting_early_stopping
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:gradient_boosting_regressor,gradient_boosting_classifier
 use sklearn_search_adapters,only:set_search_params,reset_search_clone
 use sklearn_parameter_space,only:parameter_set,param
 implicit none
 type(gradient_boosting_regressor)::reg,copy
 type(gradient_boosting_classifier)::cls,copy_cls
 type(parameter_set)::params
 real(dp)::x(43,3),y(43),w(43)
 integer::labels(43),i,j
 do i=1,43
  do j=1,3;x(i,j)=sin(real(i*j,dp));end do
  y(i)=real(i*i,dp);labels(i)=mod(i,3)*17-9;w(i)=1+.5_dp*mod(i,4)
 end do
 allocate(params%items(3))
 params%items(1)%name='n_iter_no_change';params%items(1)%value=param(3)
 params%items(2)%name='validation_fraction';params%items(2)%value=param(.25_dp)
 params%items(3)%name='tol';params%items(3)%value=param(0._dp)
 call set_search_params(reg,params);call set_search_params(cls,params)
 reg%n_estimators=30;reg%learning_rate=0;reg%subsample=.65_dp
 cls%n_estimators=30;cls%learning_rate=0;cls%subsample=.65_dp
 call reg%fit(x,y,w);call cls%fit(x,labels,w)
 call require(reg%n_estimators_==4.and.size(reg%estimators_)==4,'regression keeps the first non-improving stage')
 call require(cls%n_estimators_==4.and.size(cls%estimators_,1)==4,'classification patience counts validation stages')
 call require(size(reg%train_score_)==4.and.size(reg%oob_scores_)==4.and.size(reg%huber_delta_)==4, &
  'regression trims all learned stage histories')
 call require(size(cls%train_score_)==4.and.size(cls%oob_scores_)==4,'classification trims all learned histories')
 copy=reg;copy_cls=cls;call reset_search_clone(copy);call reset_search_clone(copy_cls)
 call require(copy%n_iter_no_change==3.and.copy%validation_fraction==.25_dp.and.copy%tol==0, &
  'regression parameter-only clone retains stopping controls')
 call require(copy_cls%n_iter_no_change==3.and.copy_cls%validation_fraction==.25_dp.and.copy_cls%tol==0, &
  'classifier parameter-only clone retains stopping controls')
 call copy%fit(x,y,w);call copy_cls%fit(x,labels,w)
 call require(all(copy%predict(x)==reg%predict(x)).and.all(copy_cls%predict(x)==cls%predict(x)), &
  'stopping clones reproduce fitted predictions')
 reg%warm_start=.true.;reg%n_estimators=4;call reg%fit(x,y,w)
 cls%warm_start=.true.;cls%n_estimators=4;call cls%fit(x,labels,w)
 call require(reg%n_estimators_==5.and.size(reg%estimators_)==4,'regression equal-count warm metadata quirk')
 call require(cls%n_estimators_==5.and.size(cls%estimators_,1)==4,'classification equal-count warm metadata quirk')
 reg%n_estimators=20;cls%n_estimators=20;call reg%fit(x,y,w);call cls%fit(x,labels,w)
 call require(size(reg%estimators_)==8.and.size(cls%estimators_,1)==8,'warm fit resets validation history')
 reg%n_iter_no_change=0;reg%n_estimators=10;call reg%fit(x,y,w)
 cls%n_iter_no_change=0;cls%n_estimators=10;call cls%fit(x,labels,w)
 call require(size(reg%estimators_)==10.and.size(cls%estimators_,1)==10,'native zero disables validation stopping')
 print *,'Classic boosting early stopping, histories, search controls, clones and warm transitions PASS'
end program
