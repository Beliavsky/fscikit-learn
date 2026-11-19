program test_adaboost
  use sklearn, only: dp,adaboost_classifier
  use sklearn_common, only: require
  implicit none
  type(adaboost_classifier) :: model
  real(dp) :: x(6,1),pair(2,1)
  real(dp), allocatable :: scores(:,:),prob(:,:),stages(:,:,:)
  integer, allocatable :: labels(:,:),prediction(:)
  x(:,1)=[-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp]
  model%n_estimators=1; call model%fit(x,[-4,-4,7,-4,7,7])
  call require(abs(model%estimator_errors_(1)-1._dp/6)<1.e-14_dp,'analytic stump error')
  call require(abs(model%estimator_weights_(1)-log(5._dp))<1.e-14_dp,'binary SAMME weight')
  scores=model%decision_function(x); prob=model%predict_proba(x)
  call require(maxval(abs(abs(scores)-2))<1.e-14_dp,'sklearn 1.6 binary decision normalization')
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-14_dp,'normalized probabilities')
  call require(abs(maxval(prob)-1/(1+exp(-2._dp)))<1.e-14_dp,'binary probability conversion')
  pair(:,1)=[-1._dp,1._dp]; model%n_estimators=8; model%learning_rate=.3_dp
  call model%fit(pair,[-4,7])
  call require(model%n_estimators_==1.and.model%estimator_weights_(1)==1,'perfect learner stops with unit weight')
  call require(sum(model%feature_importances_)==1,'stump feature importance')
  call model%fit(x,[-4,-4,7,-4,7,7],sample_weight=[1._dp,2._dp,3._dp,4._dp,5._dp,0._dp])
  scores=model%decision_function(x); stages=model%staged_decision_function(x)
  call require(maxval(abs(scores-stages(:,:,model%n_estimators_)))<1.e-14_dp,'last staged decision equals final')
  labels=model%staged_predict(x); prediction=model%predict(x)
  call require(all(labels(:,model%n_estimators_)==prediction),'last staged prediction equals final')
  stages=model%staged_predict_proba(x); prob=model%predict_proba(x)
  call require(maxval(abs(prob-stages(:,:,model%n_estimators_)))<1.e-14_dp,'last staged probability equals final')
  call require(all(model%estimator_errors_(:model%n_estimators_)<.5_dp),'accepted weak learner errors')
  print *, 'AdaBoostClassifier tests passed'
end program
