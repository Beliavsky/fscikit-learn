program test_gradient_boosting_classifier
  use sklearn
  use sklearn_common, only: require,sigmoid
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(gradient_boosting_classifier) :: model,again
  real(dp) :: x(2,1),three(3,1),expected(3,3),data(90,2),scores(3),constant(6,1)
  real(dp), allocatable :: raw(:,:),prob(:,:),logs(:,:),stages(:,:,:),last(:,:)
  integer, allocatable :: labels(:),predictions(:,:),leaves(:,:,:)
  integer :: y(90),i,j
  x(:,1)=[0._dp,1._dp]
  model%n_estimators=1; model%max_depth=1; model%learning_rate=.2_dp; model%init_zero=.true.
  call model%fit(x,[-3,9]); raw=model%decision_function(x); prob=model%predict_proba(x)
  call require(maxval(abs(raw(:,1)-[-.4_dp,.4_dp]))<1.e-14_dp,'binary Newton leaf updates')
  call require(abs(prob(2,2)-sigmoid(.4_dp))<1.e-14_dp,'binary probability link')
  call require(abs(model%train_score_(1)-2*log(1+exp(-.4_dp)))<1.e-14_dp,'binary deviance factor')
  call require(model%feature_importances_(1)==1,'normalized split improvement')
  model%loss='exponential'; call model%fit(x,[-3,9]); raw=model%decision_function(x)
  call require(maxval(abs(raw(:,1)-[-.2_dp,.2_dp]))<1.e-14_dp,'exponential Newton leaf updates')
  call require(abs(model%train_score_(1)-exp(-.2_dp))<1.e-14_dp,'exponential training loss')
  last=model%predict_proba(x)
  call require(maxval(abs(last-prob))<1.e-14_dp,'exponential half-logit probability link')
  three(:,1)=[0._dp,1._dp,2._dp]; model%loss='log_loss'; model%max_depth=2; model%learning_rate=.3_dp
  call model%fit(three,[-4,7,21]); raw=model%decision_function(three); expected=-.3_dp
  do i=1,3
    expected(i,i)=.6_dp
  end do
  call require(maxval(abs(raw-expected))<1.e-14_dp,'multiclass simultaneous gradients and Newton correction')
  call require(model%n_trees_per_iteration_==3,'one tree per class in multiclass stage')
  do i=1,90
    data(i,:)=[sin(real(i,dp)),cos(real(i*7,dp))]
    scores=[data(i,1),data(i,2),-.5_dp*sum(data(i,:))]
    j=maxloc(scores,dim=1); y(i)=-4
    if(j==2) y(i)=7
    if(j==3) y(i)=21
  end do
  model%n_estimators=25; model%max_depth=2; model%learning_rate=.2_dp; model%init_zero=.false.
  call model%fit(data,y)
  call require(model%score(data,y)>.94_dp,'multifeature multiclass recovery')
  prob=model%predict_proba(data); logs=model%predict_log_proba(data)
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-14_dp,'probability normalization')
  call require(maxval(abs(exp(logs)-prob))<1.e-14_dp,'stable log probability')
  raw=model%decision_function(data); stages=model%staged_decision_function(data)
  call require(maxval(abs(stages(:,:,25)-raw))==0,'last staged decision matches final')
  stages=model%staged_predict_proba(data)
  call require(maxval(abs(stages(:,:,25)-prob))==0,'last staged probability matches final')
  predictions=model%staged_predict(data); labels=model%predict(data)
  call require(all(predictions(:,25)==labels),'last staged labels match final')
  leaves=model%apply(data)
  call require(all(shape(leaves)==[90,25,3]).and.all(leaves>=1),'one-based leaf tensor')
  call require(all(leaves(:,1,1)==model%estimators_(1,1)%apply(data)),'leaf tensor maps to stored trees')
  call require(abs(sum(model%feature_importances_)-1)<1.e-14_dp,'multifeature importances sum to one')
  again=model; call again%fit(data,y)
  last=again%predict_proba(data)
  call require(maxval(abs(last-prob))==0,'deterministic refitting')
  constant=0; model%n_estimators=3; model%init_zero=.true.
  call model%fit(constant,[-3,-3,-3,9,9,9]); labels=model%predict(constant)
  call require(all(labels==-3).and.all(model%feature_importances_==0),'binary zero-score ties select lower class')
  model%n_estimators=1; model%learning_rate=1000; call model%fit(x,[-3,9]); logs=model%predict_log_proba(x)
  call require(all(ieee_is_finite(logs)),'log probabilities remain finite for saturated probabilities')
  print *, 'GradientBoostingClassifier tests passed'
end program
