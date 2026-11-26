program test_feature_controls
 use sklearn_common,only:dp,require
 use sklearn_tree,only:decision_tree_regressor,extra_tree_classifier
 use sklearn_ensemble,only:random_forest_classifier,extra_trees_regressor,reset_forest_fit
 use sklearn_tree_features,only:feature_count_control
 implicit none
 type(decision_tree_regressor)::tree,integer_tree
 type(extra_tree_classifier)::extra
 type(random_forest_classifier)::forest
 type(extra_trees_regressor)::reg
 type(feature_count_control)::control
 real(dp)::x(17,8),y(17)
 integer::labels(17),i,j,k
 do i=1,17
  do j=1,8;x(i,j)=sin(real(i*j,dp));end do
  y(i)=real(i*i,dp);labels(i)=mod(i,2)
 end do
 call tree%set_max_features(.5_dp);call tree%fit(x,y)
 integer_tree%max_features=4;call integer_tree%fit(x,y)
 call require(tree%max_features_==4,'fraction truncation sets learned feature budget')
 call require(all(tree%apply(x)==integer_tree%apply(x)),'fraction uses same seeded split sequence as count')
 call tree%set_max_features(1);call tree%fit(x,y);call require(tree%max_features_==1,'integer one')
 call tree%set_max_features(1._dp);call tree%fit(x,y);call require(tree%max_features_==8,'real one')
 call tree%set_max_features(11);call tree%fit(x,y);call require(tree%max_features_==11,'oversized integer is retained')
 call tree%set_max_features('log2');call tree%fit(x,y);call require(tree%max_features_==3,'log2 feature budget')
 call extra%set_max_features('none');call extra%fit(x,labels)
 call require(extra%max_features_==8,'explicit None overrides ExtraTree classifier sqrt default')
 call extra%set_max_features(0);call extra%fit(x,labels);call require(extra%max_features_==2,'integer zero restores legacy default')
 forest%warm_start=.true.;forest%n_estimators=2;call forest%set_max_features(.5_dp);call forest%fit(x,labels)
 forest%n_estimators=3;call forest%set_max_features(1);call forest%fit(x,labels)
 call require(all(forest%estimators_(:2)%max_features_==4).and.forest%estimators_(3)%max_features_==1,'warm prefix retains old feature budgets')
 call forest%set_max_features('none');call reset_forest_fit(forest);call forest%fit(x,labels)
 call require(all(forest%estimators_%max_features_==8),'clone reset preserves explicit all-feature parameter')
 reg%n_estimators=2;call reg%set_max_features(.001_dp);call reg%fit(x,y)
 call require(all(reg%estimators_%max_features_==1),'tiny fractions retain one feature')
 call control%set('log2')
 do k=0,30
  i=2**k;call require(control%resolve(0,i,.false.)==max(1,k),'exact log2 at all representable integer powers')
  if(k>0)call require(control%resolve(0,i-1,.false.)==max(1,k-1),'log2 immediately below powers')
 end do
 print *,'Typed feature budgets, default resets, oversized integers, warm prefixes and exact log2: PASS'
end program
