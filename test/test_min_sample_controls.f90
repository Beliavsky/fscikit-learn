program test_min_sample_controls
 use sklearn_common,only:dp,require
 use sklearn_tree,only:decision_tree_regressor,decision_tree_classifier,extra_tree_regressor,extra_tree_classifier
 use sklearn_ensemble,only:random_forest_regressor,random_forest_classifier,extra_trees_regressor,extra_trees_classifier, &
  gradient_boosting_regressor,gradient_boosting_classifier
 use sklearn_parameter_space,only:parameter_set,param
 use sklearn_search_adapters,only:set_search_params,fit_search_native,search_regression,search_labels,reset_search_clone
 implicit none
 class(*),allocatable::fractional,integer_model
 type(parameter_set)::parameters
 type(decision_tree_regressor)::tree,reference
 type(random_forest_regressor)::forest
 type(gradient_boosting_regressor)::booster
 type(gradient_boosting_classifier)::classifier
 real(dp)::x(25,3),y(25),w(25)
 integer::labels(25),i,j,k
 allocate(parameters%items(2))
 parameters%items(1)%name='min_samples_leaf';parameters%items(2)%name='min_samples_split'
 do i=1,25
  do j=1,3;x(i,j)=sin(real(i*j,dp));end do
  y(i)=real(i*i+mod(i,3),dp);labels(i)=mod(i,3);w(i)=1
  if(mod(i,5)==0)w(i)=0
 end do
 do k=1,10
  select case(k)
  case(1);allocate(decision_tree_regressor::fractional)
  case(2);allocate(decision_tree_classifier::fractional)
  case(3);allocate(extra_tree_regressor::fractional)
  case(4);allocate(extra_tree_classifier::fractional)
  case(5);allocate(random_forest_regressor::fractional)
  case(6);allocate(random_forest_classifier::fractional)
  case(7);allocate(extra_trees_regressor::fractional)
  case(8);allocate(extra_trees_classifier::fractional)
  case(9);allocate(gradient_boosting_regressor::fractional)
  case(10);allocate(gradient_boosting_classifier::fractional)
  end select
  parameters%items(1)%value=param(.12000000001_dp);parameters%items(2)%value=param(.4_dp)
  call set_search_params(fractional,parameters)
  allocate(integer_model,source=fractional)
  parameters%items(1)%value=param(4);parameters%items(2)%value=param(10)
  call set_search_params(integer_model,parameters)
  if(mod(k,2)==0)then
   call fit_search_native(fractional,x,yi=labels,sample_weight=w)
   call fit_search_native(integer_model,x,yi=labels,sample_weight=w)
   call require(all(search_labels(fractional,x+.137_dp)==search_labels(integer_model,x+.137_dp)), &
    'typed classifier sample fractions must use ceil(full input rows)')
  else
   call fit_search_native(fractional,x,yr=y,sample_weight=w)
   call fit_search_native(integer_model,x,yr=y,sample_weight=w)
   call require(maxval(abs(search_regression(fractional,x+.137_dp)-search_regression(integer_model,x+.137_dp)))<1.e-10_dp, &
    'typed regressor sample fractions must use ceil(full input rows)')
  end if
  call reset_search_clone(fractional)
  if(mod(k,2)==0)then
   call fit_search_native(fractional,x,yi=labels,sample_weight=w)
   call require(all(search_labels(fractional,x)==search_labels(integer_model,x)),'clone retains classifier fractions')
  else
   call fit_search_native(fractional,x,yr=y,sample_weight=w)
   call require(maxval(abs(search_regression(fractional,x)-search_regression(integer_model,x)))<1.e-10_dp, &
    'clone retains regressor fractions')
  end if
  deallocate(fractional,integer_model)
 end do
 forest%n_estimators=2;call forest%set_min_samples_leaf(.2_dp);call forest%set_min_samples_split(.6_dp)
 call forest%fit(x,y,w);tree=forest%estimators_(1);reference=tree
 call reference%set_min_samples_leaf(2);call reference%set_min_samples_split(6)
 call tree%fit(x(:10,:),y(:10));call reference%fit(x(:10,:),y(:10))
 call require(all(tree%apply(x)==reference%apply(x)),'extracted forest tree resolves fractions again on new rows')
 call tree%set_min_samples_leaf(.9_dp);call tree%set_min_samples_leaf(1)
 call tree%set_min_samples_split(1._dp);call tree%set_min_samples_split(2)
 reference=tree;call reference%fit(x,y);call tree%fit(x,y)
 call require(tree%node_count_>1.and.all(tree%apply(x)==reference%apply(x)),'integer setters reset fractional state')
 call tree%set_min_samples_leaf(.9_dp);tree%min_samples_leaf=1;call tree%fit(x,y)
 call require(tree%node_count_>1,'direct positive integer assignment overrides fractional state')
 booster%n_estimators=2;booster%warm_start=.true.
 call booster%set_min_samples_leaf(.49_dp);call booster%fit(x,y)
 call require(all(booster%estimators_%node_count_==1),'large leaf fractions constrain boosting trees')
 booster%n_estimators=4;call booster%set_min_samples_leaf(1);call booster%fit(x,y)
 call require(all(booster%estimators_(:2)%node_count_==1).and.any(booster%estimators_(3:)%node_count_>1), &
  'warm regression applies changed leaf mode only to new stages')
 classifier%n_estimators=2;classifier%warm_start=.true.
 call classifier%set_min_samples_leaf(.49_dp);call classifier%fit(x,mod(labels,2))
 call require(all(classifier%estimators_%node_count_==1),'large leaf fractions constrain binary boosting')
 classifier%n_estimators=4;call classifier%set_min_samples_leaf(1);call classifier%fit(x,mod(labels,2))
 call require(all(classifier%estimators_(:2,1)%node_count_==1).and.any(classifier%estimators_(3:,1)%node_count_>1), &
  'warm classification applies changed leaf mode only to new stages')
 print *,'Typed minimum sample controls, ten estimators, search, clones and component refits: PASS'
end program
