program count_regression_trees
  use sklearn, only: dp,decision_tree_regressor,extra_tree_regressor
  implicit none
  type(decision_tree_regressor) :: tree
  type(extra_tree_regressor) :: extra
  real(dp) :: x(6,1),y(6)
  x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp];y=[0._dp,0._dp,3._dp,4._dp,8._dp,9._dp]
  tree%criterion='poisson';tree%max_depth=2;call tree%fit(x,y)
  print *, 'Poisson predictions:',tree%predict(x)
  print *, 'Poisson node count:',tree%node_count_
  extra%criterion='friedman_mse';extra%max_depth=2;extra%random_state=7;call extra%fit(x,y)
  print *, 'Friedman ExtraTree predictions:',extra%predict(x)
end program
