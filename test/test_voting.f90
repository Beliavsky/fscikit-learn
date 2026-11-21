program test_voting
  use sklearn, only: dp,voting_classifier,voting_regressor,voting_estimator,logistic_regression,gaussian_nb, &
    dummy_classifier,linear_regression,ridge
  use sklearn_common, only: require
  implicit none
  type(voting_classifier) :: vote,saved
  type(voting_regressor) :: reg
  type(voting_estimator) :: bases(3)
  type(logistic_regression) :: lr
  type(gaussian_nb) :: nb
  type(linear_regression) :: ols
  type(ridge) :: rr
  type(dummy_classifier) :: dummy
  real(dp) :: x(8,2),yreg(8)
  integer :: y(8),i
  integer, allocatable :: pred(:)
  real(dp), allocatable :: probabilities(:,:),transform(:,:),tensor(:,:,:),prediction(:)
  x(:,1)=[(real(i,dp),i=1,8)]; x(:,2)=[0._dp,1._dp,0._dp,1._dp,0._dp,1._dp,0._dp,1._dp]
  y=[-7,-7,-7,-7,19,19,19,19]; yreg=2*x(:,1)-x(:,2)
  bases(1)%name='logistic'; allocate(bases(1)%estimator,source=lr)
  bases(2)%name='gaussian'; allocate(bases(2)%estimator,source=nb)
  bases(3)%name='dropped'; bases(3)%drop=.true.
  vote%weights=[2._dp,1._dp,100._dp]
  call vote%fit(x,y,bases); transform=vote%transform(x); pred=vote%predict(x)
  call require(size(vote%estimators_)==2.and.all(vote%active_indices_==[1,2]),'voting drops inactive estimators')
  call require(all(vote%classes_==[-7,19]).and.all(transform>=0).and.all(transform<=1),'voting label encoding')
  saved=vote; vote%voting='soft'; probabilities=vote%predict_proba(x); transform=vote%transform(x)
  tensor=vote%transform_proba(x)
  call require(all(shape(transform)==[8,4]).and.all(shape(tensor)==[2,8,2]),'voting probability transform layout')
  call require(maxval(abs(probabilities-(2*tensor(1,:,:)+tensor(2,:,:))/3))<1.e-14_dp,'voting soft weighted mean')
  vote%flatten_transform=.false.; tensor=vote%transform_proba(x)
  call require(saved%voting=='hard','voting assignment copies configuration and fitted models')
  deallocate(bases(1)%estimator,bases(2)%estimator)
  dummy%strategy='constant'; dummy%constant=0
  allocate(bases(1)%estimator,source=dummy); allocate(bases(2)%estimator,source=dummy)
  vote%voting='hard'; vote%weights=[-1._dp,-1._dp,0._dp]
  call vote%fit(x,y,bases); pred=vote%predict(x)
  call require(all(pred==-7),'hard voting respects bincount length with negative weights')
  vote%weights=0; pred=vote%predict(x)
  call require(all(pred==-7),'hard voting accepts all-zero weights and chooses first bin')
  deallocate(bases(1)%estimator,bases(2)%estimator)
  allocate(bases(1)%estimator,source=ols); allocate(bases(2)%estimator,source=rr)
  reg%weights=[1._dp,2._dp,100._dp]; transform=reg%fit_transform(x,yreg,bases)
  prediction=reg%predict(x)
  call require(maxval(abs(prediction-(transform(:,1)+2*transform(:,2))/3))<1.e-14_dp,'regression weighted mean')
  call require(reg%score(x,yreg)>.99_dp,'voting regression score')
  select type(model=>bases(1)%estimator)
  type is(linear_regression)
    call require(.not.allocated(model%coef_),'voting leaves the template unfitted')
  end select
  print *, 'voting ensembles: PASS'
end program
