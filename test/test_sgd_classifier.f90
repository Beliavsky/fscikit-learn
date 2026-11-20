program test_sgd_classifier
  use sklearn, only: dp,sgd_classifier
  use sklearn_common, only: require
  implicit none
  type(sgd_classifier) :: model,stream
  real(dp) :: x(2,1),zero(3,1)
  real(dp), allocatable :: saved(:,:),prob(:,:)
  integer, allocatable :: predicted(:)
  x(:,1)=[1._dp,2._dp]; zero=0
  model%max_iter=1; model%tol=-1; model%penalty='none'; model%learning_rate='constant'; model%eta0=.1_dp
  call model%fit(x,[-4,7])
  call require(abs(model%coef_(1,1)-.1_dp)<1.e-14_dp.and.abs(model%intercept_(1))<1.e-14_dp,'analytic hinge updates')
  call require(model%t_==3.and.model%n_iter_==1,'classifier sample clock')
  stream=sgd_classifier(); stream%penalty='none'; stream%learning_rate='constant'; stream%eta0=.1_dp
  call stream%partial_fit(x(:1,:),[-4],classes=[7,-4]); call stream%partial_fit(x(2:,:),[7])
  call require(maxval(abs(stream%coef_-model%coef_))<1.e-14_dp,'single-class batch with declared classes')
  model%loss='log_loss'; call model%fit(x,[-4,7]); saved=model%coef_
  call model%fit(x,[-4,7]); call require(all(saved==model%coef_),'fit resets state')
  model%fit_intercept=.false.; model%average=1; call model%fit(x,[-4,7]); stream=model
  stream=sgd_classifier(); stream%loss='log_loss'; stream%average=1; stream%penalty='none'
  stream%learning_rate='constant'; stream%eta0=.1_dp; stream%fit_intercept=.false.
  call stream%partial_fit(x(:1,:),[-4],classes=[-4,7]); call stream%partial_fit(x(2:,:),[7])
  call require(maxval(abs(stream%coef_-model%coef_))<1.e-14_dp,'classifier averaging across batches')
  prob=model%predict_proba(x)
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-14_dp,'binary probabilities sum to one')
  model%loss='modified_huber'; model%average=0; model%fit_intercept=.false.; call model%fit(zero,[-4,7,21])
  predicted=model%predict(zero)
  call require(all(predicted==-4),'OVR ties choose first sorted label')
  prob=model%predict_proba(zero)
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-14_dp,'multiclass probabilities sum to one')
  model%shuffle=.true.; model%random_state=42; call model%fit(x,[-4,7]); saved=model%coef_
  call model%fit(x,[-4,7]); call require(all(saved==model%coef_),'classifier native shuffle reproducibility')
  print *, 'SGDClassifier tests passed'
end program
