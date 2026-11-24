program test_tree_pruning
  use sklearn_common, only: dp,require
  use sklearn_tree, only: decision_tree_classifier,decision_tree_regressor,extra_tree_regressor,tree_pruning_path
  use sklearn_ensemble, only: random_forest_classifier,random_forest_regressor
  implicit none
  type(decision_tree_classifier) :: classifier
  type(decision_tree_regressor) :: regressor
  type(extra_tree_regressor) :: extra
  type(random_forest_classifier) :: forest
  type(random_forest_regressor) :: regression_forest
  type(tree_pruning_path) :: path
  real(dp) :: x(4,1),y(4)
  integer :: labels(4),i
  x(:,1)=[0._dp,1._dp,2._dp,3._dp];y=[0._dp,0._dp,2._dp,2._dp];labels=[-7,-7,9,9]
  path=classifier%cost_complexity_pruning_path(x,labels)
  call require(.not.allocated(classifier%nodes_),'pruning path does not fit caller')
  call require(all(path%ccp_alphas==[0._dp,.5_dp]),'Gini weakest-link alpha')
  call require(all(path%impurities==[0._dp,.5_dp]),'Gini pruning risks')
  classifier%ccp_alpha=.5_dp;call classifier%fit(x,labels)
  call require(classifier%node_count_==1.and.classifier%get_depth()==0,'inclusive alpha threshold prunes root')
  call require(all(classifier%predict(x)==-7),'pruned root retains class distribution and tie order')
  path=classifier%cost_complexity_pruning_path(x,labels)
  call require(classifier%node_count_==1.and.all(path%ccp_alphas==[0._dp,.5_dp]),'path resets clone alpha and preserves fitted caller')
  regressor%ccp_alpha=1;call regressor%fit(x,y)
  call require(regressor%node_count_==1.and.all(regressor%predict(x)==1),'pruned regression mean')
  call require(all(regressor%feature_importances_==0),'pruned stump has zero importance')
  path=regressor%cost_complexity_pruning_path(x,y)
  call require(all(path%ccp_alphas==[0._dp,1._dp]),'regression pruning path from unpruned clone')
  extra%ccp_alpha=10;extra%random_state=7;call extra%fit(x,y)
  call require(extra%node_count_==1,'ExtraTree pruning')
  path=extra%cost_complexity_pruning_path(x,y)
  call require(path%impurities(size(path%impurities))==1,'ExtraTree path ends at root risk')
  forest%n_estimators=3;forest%ccp_alpha=1;call forest%fit(x,labels)
  call require(all([(forest%estimators_(i)%node_count_==1,i=1,3)]),'classification forest forwards pruning alpha')
  regression_forest%n_estimators=3;regression_forest%ccp_alpha=10;call regression_forest%fit(x,y)
  call require(all([(regression_forest%estimators_(i)%node_count_==1,i=1,3)]),'regression forest forwards pruning alpha')
  print *, 'Cost-complexity paths, alpha boundaries, copied state and forest pruning: PASS'
end program
