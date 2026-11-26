program test_weighted_gradient_boosting_classifier
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:gradient_boosting_classifier
 use sklearn_base_dispatch,only:base_supported,base_weighted,base_proba,base_decision,base_fitted, &
  train_classifier,labels,probabilities,decisions,base_classes,seed_base
 implicit none
 type(gradient_boosting_classifier)::model,unit
 real(dp)::x(6,2),w(6)
 real(dp),allocatable::raw(:,:),stages(:,:,:),p(:,:),joint(:,:),pred(:)
 integer::y(6),i,k
 x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp];x(:,2)=mod(x(:,1),2._dp)
 y=[-9,8,25,-9,8,25];w=[1._dp,2._dp,0._dp,3._dp,2._dp,0._dp]
 model%n_estimators=4;model%max_depth=2;model%max_leaf_nodes=3
 model%ccp_alpha=.01_dp;model%min_weight_fraction_leaf=.05_dp
 call require(base_supported(model,.true.).and.base_weighted(model),'weighted classifier registry support')
 call require(base_proba(model).and.base_decision(model),'classifier probability and decision registry support')
 call require(.not.base_fitted(model),'unfitted boosting registry state')
 call seed_base(model,17);call train_classifier(model,x,y,w)
 call require(base_fitted(model).and.model%random_state==1265576559,'registry fit and NumPy-seeded random state')
 call require(model%n_classes_==3.and.all(base_classes(model)==[-9,8,25]),'zero-weight class stays in class columns')
 call require(maxval(abs(model%class_prior_-[.5_dp,.5_dp,0._dp]))<1.e-14_dp,'weighted class initialization')
 raw=model%decision_function(x);stages=model%staged_decision_function(x)
 call require(all(raw==stages(:,:,4)).and.all(raw==decisions(model,x)),'weighted staged and dispatched decision scores')
 p=model%predict_proba(x)
 call require(all(p==probabilities(model,x)).and.maxval(abs(sum(p,dim=2)-1))<1.e-14_dp,'weighted probability columns')
 call require(all(labels(model,x)==model%predict(x)),'weighted label dispatch')
 call require(abs(model%score(x,y,w)-sum(w*merge(1._dp,0._dp,model%predict(x)==y))/sum(w))<1.e-14_dp,'weighted accuracy')
 do i=1,4
  do k=1,3
   call require(model%estimators_(i,k)%get_n_leaves()<=3,'classifier leaf budget reaches stage trees')
   joint=model%estimators_(i,k)%predict_multi(x);pred=model%estimators_(i,k)%predict(x)
   call require(all(joint(:,1)==pred),'weighted Newton leaves synchronize scalar and joint storage')
  end do
 end do
 unit=model;call unit%fit(x,y)
 w=1;call model%fit(x,y,w)
 call require(all(model%decision_function(x)==unit%decision_function(x)),'explicit unit weights match unweighted classifier')
 print *,'Weighted classification boosting, retained classes, tree controls and dispatch: PASS'
end program
