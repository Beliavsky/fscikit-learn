program test_tree_absolute_error
  use sklearn_common, only: dp,require
  use sklearn_tree, only: decision_tree_regressor,extra_tree_regressor
  use sklearn_ensemble, only: random_forest_regressor,extra_trees_regressor
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(decision_tree_regressor) :: tree
  type(extra_tree_regressor) :: extra
  type(random_forest_regressor) :: forest
  type(extra_trees_regressor) :: trees
  real(dp) :: x(4,1),y(4)
  x=1;y=[0._dp,1._dp,7._dp,100._dp]
  tree%criterion='absolute_error';call tree%fit(x,y)
  call require(all(tree%predict(x)==4),'absolute-error even median differs from mean')
  call tree%fit(x,y,[0._dp,1._dp,3._dp,1._dp])
  call require(all(tree%predict(x)==7),'absolute-error weighted median')
  call tree%fit(x(:3,:),[1._dp,2._dp,3._dp],[1._dp,-1._dp,2._dp])
  call require(all(tree%predict(x)==1.5_dp),'source signed-weight median convention')
  x(:,1)=[0._dp,1._dp,2._dp,3._dp];call tree%fit(x,y)
  call require(all(tree%predict(x)==y),'absolute-error CART interpolation')
  extra%criterion='absolute_error';extra%random_state=7;call extra%fit(x,y)
  call require(all(extra%predict(x)==y),'absolute-error ExtraTree interpolation')
  forest%criterion='absolute_error';forest%n_estimators=3;forest%random_state=7
  call forest%fit(x,y);call require(all(forest%predict(x)>=0),'absolute-error bootstrapped forest')
  trees%criterion='absolute_error';trees%n_estimators=3;call trees%fit(x,y)
  call require(all(trees%predict(x)==y),'absolute-error ExtraTrees interpolation')
  call tree%fit(x,y,[0._dp,0._dp,0._dp,0._dp])
  call require(all(ieee_is_nan(tree%predict(x))),'empty median uses a deterministic quiet NaN')
  call forest%fit(x,y,[0._dp,0._dp,0._dp,0._dp])
  call require(all(ieee_is_nan(forest%predict(x))),'empty median forest uses quiet NaNs')
  call require(all(forest%feature_importances_==0),'empty median forest importance is zero')
  call trees%fit(x,y,[0._dp,0._dp,0._dp,0._dp])
  call require(all(ieee_is_nan(trees%predict(x))),'empty median ExtraTrees uses quiet NaNs')
  print *, 'Absolute-error medians, weighted splits, ExtraTrees and forests: PASS'
end program
