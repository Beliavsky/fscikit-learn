program test_adaboost_cart
  use sklearn_adaboost, only: adaboost_classifier
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  type(adaboost_classifier) :: model
  real(dp) :: x(8,2),weights(8)
  integer :: y(8),i
  real(dp), allocatable :: scores(:,:),prob(:,:),stages(:,:,:),values(:),before(:,:)
  integer, allocatable :: prediction(:)
  x(1,:)=[-1._dp,-1._dp];x(2,:)=[-1._dp,1._dp]
  x(3,:)=[1._dp,-1._dp];x(4,:)=[1._dp,1._dp];x(5:,:)=x(:4,:)
  y=[-4,7,7,-4,-4,7,7,-4];weights=[0._dp,1._dp,2._dp,3._dp,1._dp,2._dp,3._dp,4._dp]
  model%max_depth=2;model%random_state=42
  call model%fit(x,y,weights);prediction=model%predict(x)
  call require(all(prediction==y).and.model%n_estimators_==1,'depth-two CART learns XOR in a single perfect round')
  call require(model%estimators_(1)%get_depth()==2.and.size(model%estimators_)==1,'retained fitted CART objects')
  call require(model%estimator_%n_features_in_==0,'unfitted estimator prototype preserved')
  call require(abs(sum(model%feature_importances_)-1)<1.e-14_dp,'weighted CART importance normalization')
  prob=model%predict_proba(x);scores=model%predict_log_proba(x)
  call require(maxval(abs(exp(scores)-prob))<1.e-14_dp,'log probability API')
  call require(model%score(x,y,weights)==1,'weighted classification score')
  values=model%staged_score(x,y,weights)
  call require(size(values)==1.and.values(1)==1,'weighted staged classification scores')
  before=model%decision_function(x)
  call model%fit(x,y,weights);scores=model%decision_function(x)
  call require(all(scores==before),'integer random seed resets on refit')
  model%criterion='entropy';model%splitter='random';model%max_depth=8
  call model%fit(x,y);prediction=model%predict(x)
  call require(all(prediction==y),'random-split entropy CART weak learners')
  call model%fit(x,spread(21,1,8))
  scores=model%decision_function(x);prob=model%predict_proba(x)
  call require(model%n_classes_==1.and.all(scores==0).and.all(prob==1),'single-class final outputs')
  stages=model%staged_decision_function(x)
  call require(all(stages==0),'single-class staged decision correction')
  values=model%staged_score(x,spread(21,1,8))
  call require(all(values==1),'single-class staged score correction')
  model=adaboost_classifier();model%n_estimators=9;model%learning_rate=1000
  x(:,1)=[(real(i,dp),i=1,8)];x(:,2)=0;y=[-4,-4,7,-4,7,7,-4,7]
  call model%fit(x,y)
  call require(model%n_estimators_==1,'source exponential sample-weight overflow stops boosting')
  call require(ieee_is_finite(model%estimator_weights_(1)),'finite learner weight before sample-weight overflow')
  model%learning_rate=huge(1._dp);call model%fit(x,y)
  call require(.not.ieee_is_finite(model%estimator_weights_(1)),'infinite source learner weight retained quietly')
  scores=model%decision_function(x);prob=model%predict_proba(x);prediction=model%predict(x)
  call require(all(ieee_is_nan(scores)).and.all(ieee_is_nan(prob)).and.all(prediction==-4), &
    'undefined source votes preserve NaN outcomes and first-class predictions without traps')
  print *, 'AdaBoost CART weak learners: PASS'
end program
