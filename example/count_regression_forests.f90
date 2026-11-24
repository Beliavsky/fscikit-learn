program count_regression_forests
  use sklearn, only: dp,random_forest_regressor,extra_trees_regressor
  implicit none
  type(random_forest_regressor) :: forest
  type(extra_trees_regressor) :: extra
  real(dp) :: x(6,1),y(6),weights(6)
  x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp];y=[0._dp,0._dp,3._dp,4._dp,8._dp,9._dp]
  weights=[1._dp,2._dp,1._dp,0._dp,1._dp,3._dp]
  forest%n_estimators=4;forest%criterion='poisson';forest%random_state=7;forest%max_depth=2
  call forest%fit(x,y,weights)
  print *, 'Poisson forest predictions:',forest%predict(x)
  print *, 'Weighted forest R2:',forest%score(x,y,weights)
  extra%n_estimators=4;extra%criterion='friedman_mse';extra%random_state=7;extra%max_depth=2
  call extra%fit(x,y,weights)
  print *, 'Friedman ExtraTrees predictions:',extra%predict(x)
end program
