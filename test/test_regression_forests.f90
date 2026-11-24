program test_regression_forests
  use sklearn_common, only: dp,require
  use sklearn_ensemble, only: random_forest_regressor,extra_trees_regressor
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(random_forest_regressor) :: forest
  type(extra_trees_regressor) :: extra
  real(dp) :: x(8,1),y(8),w(8)
  integer :: i
  integer, allocatable :: leaves(:,:)
  x(:,1)=[(real(i,dp),i=1,8)];y=[0._dp,0._dp,1._dp,3._dp,4._dp,7._dp,8._dp,9._dp];w=1
  forest%n_estimators=3;forest%random_state=7;forest%criterion='poisson';forest%max_depth=2
  call forest%fit(x,y,w)
  call require(all([(forest%estimators_(i)%criterion=='poisson',i=1,3)]),'forest forwards Poisson criterion')
  call require(all([(forest%estimators_(i)%random_state,i=1,3)]==[327741615,976413892,1202242073]),'source forest tree seeds')
  call require(all(forest%predict(x)>0),'Poisson forest positive predictions')
  leaves=forest%apply(x);call require(all(shape(leaves)==[8,3]),'forest leaf matrix')
  call require(forest%n_features_in_==1,'forest learned feature count')
  forest%max_samples=2;call forest%fit(x,y)
  call require(all([(forest%estimators_(i)%nodes_(1)%weighted_n_samples==2,i=1,3)]),'max_samples bootstrap mass')
  call require(all([(forest%estimators_(i)%nodes_(1)%n_samples<=2,i=1,3)]),'bootstrap cardinality counts unique active rows')
  extra%n_estimators=3;extra%criterion='friedman_mse';extra%random_state=7
  call extra%fit(x,y,w)
  call require(all(extra%predict(x)==y),'Friedman ExtraTrees interpolation')
  call require(extra%score(x,y,w)==1,'weighted forest R2')
  w=0;call extra%fit(x,y,w)
  call require(all(ieee_is_nan(extra%predict(x))),'zero-weight forest preserves source NaN predictions')
  call require(all(extra%feature_importances_==0),'all-stump forest returns zero importances')
  print *, 'Regression forests, source bootstraps, weighted criteria and leaf matrices: PASS'
end program
