program test_recursive_selection
  use sklearn, only: dp,rfe,rfecv,linear_regression,ridge,logistic_regression
  use sklearn_common, only: require
  implicit none
  type(rfe) :: selector
  type(rfecv) :: search
  type(linear_regression) :: ols
  type(ridge) :: regularized
  type(logistic_regression) :: classifier
  real(dp) :: x(48,5),y(48)
  real(dp), allocatable :: z(:,:),restored(:,:),prediction(:),prob(:,:)
  integer :: i,labels(48)
  integer, allocatable :: classes(:)
  logical, allocatable :: mask(:)
  do i=1,48
    x(i,:)=[sin(real(i,dp)),cos(real(i*2,dp)),sin(real(i*3,dp)),cos(real(i*4,dp)),sin(real(i*5,dp))]
  end do
  y=3*x(:,2)-2*x(:,4); labels=merge(9,-4,y>0)
  selector%n_features_to_select=2; call selector%fit(x,y,ols)
  mask=selector%get_support()
  call require(all(mask.eqv.[.false.,.true.,.false.,.true.,.false.]),'RFE informative features')
  call require(all(pack(selector%ranking_,mask)==1),'selected ranks equal one')
  prediction=selector%predict(x); call require(maxval(abs(prediction-y))<1.e-10_dp,'RFE predicts with reduced model')
  call require(selector%score(x,y)>1-1.e-12_dp,'RFE regression score')
  z=selector%transform(x); restored=selector%inverse_transform(z)
  call require(all(restored(:,2)==x(:,2)).and.all(restored(:,1)==0),'RFE inverse zero fills')
  call require(.not.allocated(ols%coef_),'RFE leaves caller estimator unfitted')
  selector%step=.5_dp; selector%n_features_to_select=0; call selector%fit(x,y,ols)
  call require(selector%n_features_==2.and.maxval(selector%ranking_)==3,'fractional elimination width')
  selector%step=100; selector%n_features_to_select=10; call selector%fit(x,y,ols)
  call require(selector%n_features_==5.and.all(selector%ranking_==1),'oversized count keeps all features')
  regularized%alpha=.1_dp; search%cv=4; call search%fit(x,y,regularized)
  call require(search%n_features_>=2.and.search%score(x,y)>.999_dp,'RFECV regression fitted predictions')
  call require(all(search%cv_results_%n_features==[1,2,3,4,5]),'ascending RFECV counts')
  call require(all(search%cv_results_%std_test_score>=0),'nonnegative CV deviations')
  call require(size(search%cv_results_%split_test_score,2)==4,'per-fold score layout')
  selector%n_features_to_select=2; selector%step=1
  classifier%tol=1.e-10_dp; call selector%fit(x,labels,classifier)
  classes=selector%predict_classes(x); prob=selector%predict_proba(x)
  call require(all(abs(sum(prob,dim=2)-1)<1.e-12_dp),'RFE class probabilities')
  call require(selector%score(x,labels)>.9_dp,'RFE classifier score')
  call require(.not.allocated(selector%regression_estimator_),'refit clears old estimator kind')
  call require(.not.allocated(classifier%coef_),'RFE leaves caller classifier unfitted')
  search%step=2; search%min_features_to_select=2; call search%fit(x,labels,classifier)
  call require(all(search%cv_results_%n_features==[2,3,5]),'RFECV nondivisible elimination path')
  call require(search%score(x,labels)>.9_dp,'RFECV classifier prediction')
  search%min_features_to_select=10; call search%fit(x,y,ols)
  call require(search%n_features_==5.and.size(search%cv_results_%n_features)==1,'RFECV all-feature boundary')
  search=rfecv(); search%cv=4; y=0
  z=search%fit_transform(x,y,ols)
  call require(search%n_features_==1,'exact CV score ties select fewest features')
  call require(all(search%cv_results_%mean_test_score==1),'constant response CV tie fixture')
  call require(search%support_(5).and.all(search%ranking_==[5,4,3,2,1]),'coefficient ties eliminate first column')
  print *, 'RFE and RFECV tests passed'
end program
