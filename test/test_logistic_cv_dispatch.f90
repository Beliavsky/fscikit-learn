program test_logistic_cv_dispatch
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(logistic_regression_cv) :: base
  type(gaussian_nb) :: nb
  type(voting_estimator) :: bases(2)
  type(voting_classifier) :: vote
  type(bagging_classifier) :: bag
  type(stacking_classifier) :: stack
  type(one_vs_rest_classifier) :: ovr
  type(multi_output_classifier) :: multi
  type(calibrated_classifier_cv) :: calibrated
  type(self_training_classifier) :: semi
  type(multioutput_probability), allocatable :: outputs(:)
  real(dp) :: x(72,2),w(72)
  integer :: binary(72),many(72),targets(72,2),unlabeled(72),i
  real(dp), allocatable :: prob(:,:),scores(:)
  do i=1,72
    x(i,:)=[sin(real(i,dp)*.7_dp),cos(real(i,dp)*.31_dp)]
    binary(i)=merge(-3,7,mod(i,2)==0); many(i)=10*mod(i,3)+4; w(i)=1+.1_dp*mod(i,4)
  end do
  targets(:,1)=binary; targets(:,2)=many; unlabeled=binary; unlabeled(1::4)=-1
  base%cv=2; base%cs=[.1_dp,1._dp]; base%tol=1.e-10_dp; base%max_iter=1000
  scores=cross_val_score(base,x,binary,cv=3)
  call require(size(scores)==3.and.all(scores>=0.and.scores<=1),'nested logistic CV cross_val_score')
  bases(1)%name='lrcv'; allocate(bases(1)%estimator,source=base)
  bases(2)%name='nb'; allocate(bases(2)%estimator,source=nb)
  vote%voting='soft'; call vote%fit(x,binary,bases,sample_weight=w); prob=vote%predict_proba(x(:4,:))
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-13_dp,'CV logistic voting probabilities')
  select type(model=>vote%estimators_(1)%estimator)
  type is(logistic_regression_cv)
    call require(allocated(model%coefs_paths_),'voting preserves CV dynamic type and paths')
  class default
    call require(.false.,'voting lost LogisticRegressionCV subtype')
  end select
  bag%n_estimators=2; bag%random_state=7; call bag%fit(x,binary,base,sample_weight=w)
  prob=bag%predict_proba(x(:4,:)); call require(all(prob>=0.and.prob<=1),'CV logistic bagging')
  stack%cv=3; call stack%fit(x,binary,bases,final_estimator=base,sample_weight=w)
  prob=stack%predict_proba(x(:4,:)); call require(maxval(abs(sum(prob,dim=2)-1))<1.e-13_dp,'CV logistic stacking')
  call ovr%fit(x,many,base,sample_weight=w); prob=ovr%predict_proba(x(:4,:))
  call require(all(shape(prob)==[4,3]),'CV logistic multiclass reduction')
  call multi%fit(x,targets,base,sample_weight=w); outputs=multi%predict_proba(x(:4,:))
  call require(size(outputs)==2.and.size(outputs(2)%values,2)==3,'CV logistic independent outputs')
  calibrated%cv=2; call calibrated%fit(x,binary,base,sample_weight=w); prob=calibrated%predict_proba(x(:4,:))
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-13_dp,'CV logistic calibration')
  semi%max_iter=1; semi%criterion='k_best'; semi%k_best=4
  call semi%fit(x,unlabeled,base); prob=semi%predict_proba(x(:4,:))
  call require(count(semi%transduction_/=-1)==58,'CV logistic self-training fit and final refit')
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-13_dp,'CV logistic self-training probabilities')
  print *, 'LogisticRegressionCV composite dispatch: PASS'
end program
