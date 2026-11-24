program test_tree_regression_criteria
  use sklearn_tree, only: decision_tree_regressor,extra_tree_regressor
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  type(decision_tree_regressor) :: model,squared
  type(extra_tree_regressor) :: extra
  real(dp) :: x(4,1),y(4),query(2,1)
  real(dp), allocatable :: prediction(:)
  x(:,1)=[0._dp,1._dp,2._dp,3._dp];y=[0._dp,0._dp,2._dp,2._dp]
  model%criterion='friedman_mse';model%min_impurity_decrease=2
  call model%fit(x,y)
  call require(model%node_count_==3,'Friedman improvement uses source unnormalized SSE scale')
  call require(all(model%predict(x)==y),'Friedman leaf predictions are weighted means')
  squared%min_impurity_decrease=2;call squared%fit(x,y)
  call require(squared%node_count_==1,'Squared-error threshold remains globally normalized')
  model=decision_tree_regressor();model%criterion='poisson'
  call model%fit(x(:2,:),[0._dp,1._dp])
  call require(model%node_count_==1,'Poisson forbids a child with zero weighted target sum')
  call require(all(model%predict(x(:2,:))==.5_dp),'Poisson zero-count branch remains in a positive leaf')
  call model%fit(x,[1._dp,2._dp,4._dp,8._dp],sample_weight=[1._dp,2._dp,0._dp,1._dp])
  prediction=model%predict(x)
  call require(all(prediction>0).and.all(ieee_is_finite(prediction)),'Poisson weighted positive predictions')
  call require(model%nodes_(1)%n_samples==3,'Poisson omits zero-weight training rows')
  extra%criterion='poisson';extra%random_state=7;extra%max_depth=2
  call extra%fit(x,[1._dp,2._dp,4._dp,8._dp])
  call require(all(extra%predict(x)>0),'ExtraTree uses Poisson criterion')
  extra%criterion='friedman_mse';call extra%fit(x,y)
  call require(extra%node_count_>=3,'ExtraTree uses Friedman criterion')
  call model%fit(x,[1._dp,2._dp,4._dp,8._dp],sample_weight=[0._dp,0._dp,0._dp,0._dp])
  call require(all(ieee_is_nan(model%predict(x))),'Zero-weight Poisson root preserves source NaN mean')
  print *, 'Poisson/Friedman regression trees, randomized splits and weighted leaves: PASS'
end program
