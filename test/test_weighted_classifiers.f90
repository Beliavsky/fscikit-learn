program test_weighted_classifiers
  use sklearn, only: dp,gaussian_nb,logistic_regression,calibrated_classifier_cv,self_training_classifier
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  type(gaussian_nb) :: nb,batch,copy
  type(logistic_regression) :: lr,replicated,small
  type(calibrated_classifier_cv) :: calibration
  type(self_training_classifier) :: self_model
  real(dp) :: x(6,1),query(3,1),w(6)=[1._dp,2._dp,1._dp,2._dp,1._dp,1._dp],zeros(6,1)=0
  real(dp), allocatable :: prob(:,:),logs(:,:)
  integer :: y(6)=[0,0,0,1,1,1]
  x(:,1)=[-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp]; query(:,1)=[-2._dp,0._dp,2._dp]
  call nb%fit(x,y,w)
  call require(all(nb%class_count_==[4._dp,4._dp]),'weighted Gaussian class mass')
  call require(maxval(abs(nb%theta_(:,1)-[-2._dp,1.75_dp]))<1.e-14_dp,'weighted Gaussian mean')
  call require(maxval(abs(nb%var_(:,1)-nb%epsilon_-[.5_dp,.6875_dp]))<1.e-14_dp,'weighted population variance')
  nb%var_smoothing=0
  call nb%partial_fit(x(:2,:),y(:2),[1,0,1],w(:2))
  call require(all(nb%classes_==[0,1]),'repeated class declaration has the same sorted class set')
  call batch%fit(x,y,w); copy=batch
  call require(all(copy%class_count_==batch%class_count_),'deep-copy Gaussian statistics')
  nb=gaussian_nb(); nb%var_smoothing=0
  call nb%partial_fit(x(:2,:),y(:2),[1,0,1],w(:2))
  call require(all(nb%class_count_==[3._dp,0._dp]),'declared unseen class has zero mass')
  call nb%partial_fit(x(3:,:),y(3:),sample_weight=w(3:))
  batch%var_smoothing=0; call batch%fit(x,y,w)
  call require(maxval(abs(nb%theta_-batch%theta_))<1.e-14_dp,'online weighted means match pooled means')
  call require(maxval(abs(nb%var_-batch%var_))<1.e-14_dp,'online Chan population variance')
  copy=nb; call nb%partial_fit(x,y,sample_weight=0*w)
  call require(all(nb%class_count_==copy%class_count_).and.all(nb%theta_==copy%theta_), &
    'zero-weight streaming batch preserves mass and means')
  call nb%fit(x,y,w*1.e-12_dp)
  call require(all(nb%theta_==0).and.all(nb%var_==0),'source tiny-mass moment skip')
  call require(all(nb%class_count_>0),'tiny skipped moments still increase class counts')
  call nb%fit(x,y,0*w)
  prob=nb%predict_proba(query)
  call require(all(ieee_is_nan(prob)),'zero-total empirical Gaussian posterior is undefined')
  call require(all(nb%predict(query)==0),'undefined joint likelihood chooses first NaN class')
  call nb%fit(reshape([1._dp,3._dp],[2,1]),[2,2],[5.e-9_dp,5.e-9_dp])
  call require(all(nb%theta_==0),'source near-zero mass cutoff is inclusive')
  call nb%fit(reshape([1._dp,3._dp],[2,1]),[2,2],[5.1e-9_dp,5.1e-9_dp])
  call require(abs(nb%theta_(1,1)-2)<1.e-14_dp,'moment updates resume above the source cutoff')
  nb%priors=[0._dp,1._dp]; nb%var_smoothing=1.e-9_dp; call nb%fit(x,y,w)
  prob=nb%predict_proba(query); logs=nb%predict_log_proba(query)
  call require(all(prob(:,1)==0).and.all(prob(:,2)==1),'fixed zero prior excludes a class')
  call require(all(.not.ieee_is_finite(logs(:,1))),'zero prior has negative infinite log posterior')
  nb=gaussian_nb(); nb%var_smoothing=0
  call nb%fit(reshape([-2._dp,-1._dp,1._dp,1._dp],[4,1]),[0,0,1,1])
  call require(all(nb%predict(query)==1),'a later invalid Gaussian class wins source NaN argmax')
  call lr%fit(x,y,w)
  call replicated%fit(x([1,2,2,3,4,4,5,6],:),y([1,2,2,3,4,4,5,6]))
  call require(maxval(abs(lr%coef_-replicated%coef_))<1.e-8_dp,'integer weights equal sample replication')
  call require(maxval(abs(lr%intercept_-replicated%intercept_))<1.e-8_dp,'weighted logistic intercept')
  call small%fit(zeros,y,[1._dp,1._dp,1._dp,3._dp,3._dp,3._dp]*1.e-100_dp)
  prob=small%predict_proba(0*query)
  call require(maxval(abs(prob(:,2)-.75_dp))<1.e-9_dp,'tiny weight mass still fits the logistic intercept')
  small%c=1.e-310_dp
  call small%fit(zeros,y,[1._dp,1._dp,1._dp,3._dp,3._dp,3._dp]*1.e150_dp)
  prob=small%predict_proba(0*query)
  call require(maxval(abs(prob(:,2)-.75_dp))<1.e-9_dp,'weighted C scaling avoids an overflowing reciprocal')
  small%fit_intercept=.false.; call small%fit(x,y,w*1.e150_dp)
  prob=small%predict_proba(query)
  call require(all(prob==.5_dp),'extreme regularization rounds probabilities to one half')
  call require(all(small%predict(query)==[0,0,1]),'logistic classification uses raw scores before probability rounding')
  small%c=1.e-100_dp
  call self_model%fit(x,[0,0,-1,-1,1,1],small)
  call require(all(self_model%predict(query)==[0,0,1]),'self-training delegates raw-score classifier prediction')
  calibration%cv=2
  call calibration%fit(x,y,lr,w)
  prob=calibration%predict_proba(query)
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-14_dp,'weighted logistic CV calibration')
  nb=gaussian_nb(); call calibration%fit(x,y,nb,w)
  prob=calibration%predict_proba(query)
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-14_dp,'weighted Gaussian CV calibration')
  print *, 'weighted and streaming classifiers: PASS'
end program
