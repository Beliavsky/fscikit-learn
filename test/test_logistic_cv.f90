program test_logistic_cv
  use sklearn_linear_model, only: logistic_regression_cv,logistic_regression
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(logistic_regression_cv) :: cv,copy
  type(logistic_regression) :: base
  real(dp) :: x(60,2),w(60),t
  integer :: y(60),i,f,j
  real(dp), allocatable :: prob(:,:),average(:,:),score_sums(:)
  do i=1,60
    t=real(i,dp)
    x(i,:)=[sin(.7_dp*t),cos(1.1_dp*t)]
    y(i)=merge(7,-3,mod(i,3)==0.or.x(i,1)>.7_dp); w(i)=1+mod(i,4)*.25_dp
  end do
  cv%cs=[.01_dp,.1_dp,1._dp,10._dp]; cv%cv=3; cv%tol=1.e-10_dp; cv%max_iter=500
  call cv%fit(x,y,w)
  call require(all(cv%classes_==[-3,7]).and.all(cv%path_classes_==[7]),'LogisticRegressionCV path class order')
  call require(all(shape(cv%coefs_paths_)==[1,3,4,3]),'LogisticRegressionCV coefficient path layout')
  call require(all(shape(cv%scores_)==[3,4]),'LogisticRegressionCV joint score layout')
  score_sums=sum(cv%scores_,dim=1); j=maxloc(score_sums,dim=1)
  call require(cv%c_(1)==cv%cs_(j),'LogisticRegressionCV global score selection')
  average=sum(cv%coefs_paths_(:,:,j,:),dim=2)/3
  call require(maxval(abs(average-cv%refit_init_))<1.e-14_dp,'LogisticRegressionCV average refit initialization')
  base%c=cv%c_(1); base%tol=cv%tol; base%max_iter=cv%max_iter; call base%fit(x,y,w)
  call require(maxval(abs(base%coef_-cv%coef_))<1.e-8_dp,'LogisticRegressionCV standalone final equivalence')
  call require(abs(base%score(x,y,w)-cv%score(x,y,w))<1.e-14_dp,'weighted logistic score')
  copy=cv; cv%refit=.false.; call cv%fit(x,y,w)
  average=0
  do f=1,3
    j=maxloc(cv%scores_(f,:),dim=1); average=average+cv%coefs_paths_(:,f,j,:)/3
  end do
  call require(maxval(abs(average(:,:2)-cv%coef_))<1.e-14_dp,'LogisticRegressionCV refit=false fold average')
  call require(cv%n_iter_==0.and.copy%n_iter_>0,'LogisticRegressionCV final versus path iteration state')
  cv=copy; cv%class_weight_balanced=.true.; cv%scoring='neg_log_loss'; call cv%fit(x,y,w)
  call require(abs(cv%class_weight_(1)-60._dp/(2*count(y==-3)))<1.e-14_dp,'global balanced class weights')
  call require(cv%score(x,y,w)<0,'LogisticRegressionCV configured scoring on queries')
  cv=copy; cv%fit_intercept=.false.; cv%scoring='roc_auc'
  allocate(cv%splits(1)); cv%splits(1)%train=pack([(i,i=1,60)],y==-3); cv%splits(1)%test=[1,2]
  call cv%fit(x,y)
  call require(all(cv%intercept_==0),'LogisticRegressionCV binary single-class fold without intercept')
  cv%splits(1)%test=pack([(i,i=1,60)],y==-3); call cv%fit(x,y)
  call require(all(ieee_is_nan(cv%scores_)).and.cv%c_(1)==cv%cs_(1), &
    'LogisticRegressionCV ROC single-class test NaNs select first C')
  cv=copy; deallocate(cv%cs); cv%n_cs=1; call cv%fit(x,y)
  call require(size(cv%cs_)==1.and.cv%cs_(1)==1.e-4_dp,'LogisticRegressionCV one-point automatic C grid')
  cv=copy; cv%max_iter=0; call cv%fit(x,y)
  call require(all(cv%coef_==0).and.all(cv%intercept_==0).and.all(cv%n_iter_path_==0), &
    'LogisticRegressionCV zero-iteration paths and refit')
  cv=copy; cv%max_iter=1; call cv%fit(x,y)
  call require(all(cv%n_iter_path_==1).and..not.cv%all_cv_converged_,'LogisticRegressionCV honors iteration cap')
  print *, 'LogisticRegressionCV: PASS'
end program
