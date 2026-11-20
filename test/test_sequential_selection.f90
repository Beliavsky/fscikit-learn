program test_sequential_selection
  use sklearn, only: dp,sequential_feature_selector,linear_regression,ridge
  use sklearn_common, only: require
  implicit none
  type(sequential_feature_selector) :: model
  type(linear_regression) :: estimator
  type(ridge) :: regularized
  real(dp) :: x(40,5),y(40)
  real(dp), allocatable :: z(:,:),restored(:,:)
  logical, allocatable :: mask(:)
  integer :: i
  do i=1,40
    x(i,:)=[sin(real(i,dp)),cos(real(i*2,dp)),sin(real(i*3,dp)),cos(real(i*4,dp)),sin(real(i*5,dp))]
  end do
  y=3*x(:,2)-2*x(:,4); model%n_features_to_select=2
  z=model%fit_transform(x,y,estimator); mask=model%get_support()
  call require(all(mask.eqv.[.false.,.true.,.false.,.true.,.false.]),'forward selects informative features')
  restored=model%inverse_transform(z)
  call require(all(restored(:,2)==x(:,2)).and.all(restored(:,4)==x(:,4)),'inverse restores selected columns')
  call require(all(restored(:,1)==0).and.all(restored(:,3)==0),'inverse fills missing columns')
  model%direction='backward'; call model%fit(x,y,estimator)
  call require(all(model%support_.eqv.mask),'backward selects informative features')
  call require(.not.allocated(estimator%coef_),'selector leaves caller estimator unfitted')
  model%n_features_to_select=0; call model%fit(x,y,estimator)
  call require(model%n_features_to_select_==3,'odd backward auto selects ceil half')
  model%direction='forward'; call model%fit(x,y,estimator)
  call require(model%n_features_to_select_==2,'forward auto selects floor half')
  model%use_tol=.true.; model%tol=1.e6_dp; call model%fit(x,y,estimator)
  call require(model%n_features_to_select_==1,'forward tolerance stops after first addition')
  model%direction='backward'; call model%fit(x,y,estimator)
  call require(model%n_features_to_select_==4,'backward tolerance stops after first removal')
  model%use_tol=.false.; model%n_features_to_select=2; regularized%alpha=.1_dp
  call model%fit(x,y,regularized)
  call require(all(model%support_.eqv.mask),'ridge polymorphic estimator')
  print *, 'SequentialFeatureSelector tests passed'
end program
