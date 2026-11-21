program test_calibrated
  use sklearn, only: dp,calibrated_classifier_cv,logistic_regression,gaussian_nb
  use sklearn_common, only: require
  implicit none
  type(calibrated_classifier_cv) :: model,copy
  type(logistic_regression) :: lr
  type(gaussian_nb) :: nb,full
  real(dp) :: zero(4,1)=0,query(3,1)=0,x(20,2),query2(2,2)
  real(dp), allocatable :: prob(:,:),before(:,:),after(:,:)
  integer :: y(20),i
  call lr%fit(zero,[0,1,0,1]); before=lr%predict_proba(query)
  model%prefit=.true.
  call model%fit(zero,[0,0,0,1],lr)
  prob=model%predict_proba(query)
  call require(maxval(abs(prob(:,2)-19._dp/60))<1.e-10_dp,'constant sigmoid smoothed target mean')
  call require(model%score(query,[0,0,0])==1,'calibrated accuracy')
  call require(.not.allocated(model%fold_id_),'prefit has no CV assignment')
  copy=model
  call model%fit(zero,[1,1,1,0],lr)
  prob=model%predict_proba(query)
  call require(maxval(abs(prob(:,2)-41._dp/60))<1.e-10_dp,'complemented sigmoid targets')
  prob=copy%predict_proba(query)
  call require(maxval(abs(prob(:,2)-19._dp/60))<1.e-10_dp,'deep-copy calibrated state')
  call model%fit(zero,[0,0,0,1],lr,[1._dp,2._dp,0._dp,3._dp])
  prob=model%predict_proba(query)
  call require(maxval(abs(prob-.5_dp))<1.e-10_dp,'weighted sigmoid priors and zero weights')
  call model%fit(zero,[0,0,0,1],lr,[0._dp,0._dp,0._dp,0._dp])
  prob=model%predict_proba(query)
  call require(all(prob==.5_dp),'zero total sigmoid calibration weight')
  model%method='isotonic'; call model%fit(zero,[0,0,0,1],lr)
  prob=model%predict_proba(query)
  call require(maxval(abs(prob(:,2)-.25_dp))<1.e-14_dp,'constant isotonic empirical target mean')
  after=lr%predict_proba(query)
  call require(all(before==after),'prefit estimator preserved')
  do i=1,20
    select case(mod(i-1,3))
    case(0)
      y(i)=11
    case(1)
      y(i)=-7
    case(2)
      y(i)=4
    end select
    x(i,:)=[sin(real(i,dp)),cos(real(2*i,dp))]+real(mod(i,3),dp)
  end do
  query2=x([2,13],:); model%prefit=.false.; model%cv=3; model%ensemble=.true.
  call model%fit(x,y,nb)
  call require(size(model%calibrated_classifiers_)==3,'one fitted pair per ensemble fold')
  call require(all(model%fold_id_(:7)==1).and.all(model%fold_id_(8:14)==2).and. &
    all(model%fold_id_(15:)==3),'first-appearance class encoding in stratified CV')
  prob=model%predict_proba(query2)
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-14_dp.and.minval(prob)>=0,'normalized multiclass probabilities')
  model%ensemble=.false.; call model%fit(x,y,nb)
  call require(size(model%calibrated_classifiers_)==1,'single out-of-fold calibration pair')
  call full%fit(x,y)
  select type(base=>model%calibrated_classifiers_(1)%estimator_)
  type is(gaussian_nb)
    call require(all(base%theta_==full%theta_).and.all(base%var_==full%var_),'full-data base refit after OOF scores')
  class default
    call require(.false.,'wrong calibrated estimator type')
  end select
  call lr%fit(x,y)
  call require(abs(sum(lr%intercept_))<1.e-12_dp,'canonical multiclass softmax intercept gauge')
  model%prefit=.true.; call model%fit(x,0*y+99,lr)
  prob=model%predict_proba(query2)
  call require(maxval(abs(prob-1._dp/3))<1.e-14_dp,'zero multiclass calibration mass becomes uniform')
  call require(all(model%classes_==[-7,4,11]),'prefit retains estimator class universe')
  call require(.not.allocated(model%fold_id_),'refit clears obsolete fold state')
  print *, 'calibrated classifiers: PASS'
end program
