program test_adaboost_tree_controls
 use sklearn_common,only:dp,require
 use sklearn_adaboost,only:adaboost_classifier
 use sklearn_adaboost_regression,only:adaboost_regressor
 use sklearn_tree,only:decision_tree_regressor,decision_tree_classifier
 use sklearn_base_dispatch,only:seed_base,base_supported,base_weighted,base_fitted
 use sklearn_parameter_space,only:parameter_set,param
 use sklearn_search_adapters,only:set_search_params,reset_search_clone
 implicit none
 type(adaboost_classifier)::cls,cls_copy
 type(adaboost_regressor)::reg,reg_copy
 type(decision_tree_regressor)::component,reference
 type(decision_tree_classifier)::class_component,class_reference
 type(parameter_set)::params
 real(dp)::x(37,3),y(37),w(37),new_x(10,6),new_y(10)
 integer::labels(37),new_labels(10),i,j
 allocate(params%items(5))
 params%items(1)%name='estimator__min_samples_leaf';params%items(1)%value=param(.1_dp)
 params%items(2)%name='estimator__min_samples_split';params%items(2)%value=param(.3_dp)
 params%items(3)%name='estimator__max_features';params%items(3)%value=param(.7_dp)
 params%items(4)%name='estimator__max_leaf_nodes';params%items(4)%value=param(4)
 params%items(5)%name='estimator__ccp_alpha';params%items(5)%value=param(.001_dp)
 do i=1,37
  do j=1,3;x(i,j)=sin(real(i*j,dp));end do
  y(i)=real(i*i,dp);labels(i)=mod(i,3)*17-9;w(i)=1+.5_dp*mod(i,4)
  if(mod(i,7)==0)w(i)=0
 end do
 call set_search_params(reg,params);call set_search_params(cls,params)
 reg%n_estimators=4;cls%n_estimators=4
 call reg%fit(x,y,w);call cls%fit(x,labels,w)
 call require(reg%estimator_%node_count_==0.and.reg%estimator_%n_features_in_==0,'regression prototype remains unfitted')
 call require(cls%estimator_%node_count_==0.and.cls%estimator_%n_features_in_==0,'classifier prototype remains unfitted')
 call require(all(reg%estimators_%max_features_==2).and.all(cls%estimators_%max_features_==2),'typed feature budgets reach every round')
 do i=1,10
  do j=1,6;new_x(i,j)=cos(real(i*j,dp));end do
  new_y(i)=real(i*i,dp);new_labels(i)=mod(i,2)
 end do
 component=reg%estimators_(1);reference=component
 call reference%set_min_samples_leaf(1);call reference%set_min_samples_split(3);call reference%set_max_features(4)
 call component%fit(new_x,new_y);call reference%fit(new_x,new_y)
 call require(component%max_features_==4.and.all(component%apply(new_x)==reference%apply(new_x)), &
  'regression component retains real controls and resolves new row/feature counts')
 class_component=cls%estimators_(1);class_reference=class_component
 call class_reference%set_min_samples_leaf(1);call class_reference%set_min_samples_split(3);call class_reference%set_max_features(4)
 call class_component%fit(new_x,new_labels);call class_reference%fit(new_x,new_labels)
 call require(class_component%max_features_==4.and.all(class_component%apply(new_x)==class_reference%apply(new_x)), &
  'classification component retains real controls on refit')
 reg_copy=reg;cls_copy=cls;call reset_search_clone(reg_copy);call reset_search_clone(cls_copy)
 call reg_copy%fit(x,y,w);call cls_copy%fit(x,labels,w)
 call require(all(reg_copy%predict(x)==reg%predict(x)).and.all(cls_copy%predict(x)==cls%predict(x)), &
  'fitted template clones refit independently with retained controls')
 call reg%set_min_samples_leaf(1);call reg%set_min_samples_split(2);call reg%set_max_features(1._dp)
 call cls%set_min_samples_leaf(1);call cls%set_min_samples_split(2);call cls%set_max_features(1._dp)
 call reg%fit(x,y);call cls%fit(x,labels)
 call require(all(reg%estimators_%max_features_==3).and.all(cls%estimators_%max_features_==3),'real one selects all features')
 call require(base_supported(reg,.false.).and.base_weighted(reg).and.base_fitted(reg),'AdaBoost.R2 registry capabilities')
 call seed_base(reg,42)
 call require(reg%tree_random_state==1608637542.and.reg%random_state==1273642419,'R2 nested seeds follow Source sorted order')
 print *,'AdaBoost typed controls, pruning, complete shared trees, refits and registry integration PASS'
end program
