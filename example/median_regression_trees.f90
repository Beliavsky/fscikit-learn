program median_regression_trees
  use sklearn, only: dp,decision_tree_regressor,random_forest_regressor
  implicit none
  type(decision_tree_regressor) :: median_tree,mean_tree
  type(random_forest_regressor) :: forest
  real(dp) :: x(4,1),y(4),weights(4)
  x=1;y=[0._dp,1._dp,7._dp,100._dp]
  median_tree%criterion='absolute_error';call median_tree%fit(x,y);call mean_tree%fit(x,y)
  print *, 'Median leaf:',median_tree%predict(x(:1,:))
  print *, 'Mean leaf:',mean_tree%predict(x(:1,:))
  x(:,1)=[0._dp,1._dp,2._dp,3._dp];weights=[1._dp,2._dp,1._dp,1._dp]
  forest%criterion='absolute_error';forest%n_estimators=4;forest%random_state=7;forest%max_depth=2
  call forest%fit(x,y,weights)
  print *, 'Weighted median forest:',forest%predict(x)
end program
