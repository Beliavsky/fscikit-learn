program test_adaboost_regression
  use sklearn, only: dp,adaboost_regressor
  use sklearn_common, only: require
  implicit none
  type(adaboost_regressor) :: model
  real(dp) :: x(20,2),y(20),w(20),pred(20)
  real(dp), allocatable :: stages(:,:),scores(:)
  integer :: i,k
  do i=1,20
    x(i,:)=[real(i,dp)/10,sin(real(i,dp))]; y(i)=x(i,1)**2+x(i,2)
  end do
  w=1; w(1)=0; model%n_estimators=8; model%random_state=42
  do k=1,3
    select case(k)
    case(1); model%loss='linear'
    case(2); model%loss='square'
    case(3); model%loss='exponential'
    end select
    call model%fit(x,y,w)
    stages=model%staged_predict(x); pred=model%predict(x); scores=model%staged_score(x,y)
    call require(maxval(abs(pred-stages(:,model%n_estimators_)))<1.e-14_dp,'AdaBoost final stage')
    call require(abs(scores(model%n_estimators_)-model%score(x,y))<1.e-14_dp,'AdaBoost staged score')
    call require(model%score(x,y)>.8_dp,'AdaBoost nonlinear fit')
    call require(abs(sum(model%feature_importances_)-1)<1.e-12_dp,'AdaBoost importances')
    call require(all(model%estimator_errors_(:model%n_estimators_)<.5_dp),'AdaBoost accepted errors')
  end do
  call model%fit(x,spread(4._dp,1,20))
  call require(model%n_estimators_==1.and.model%estimator_weights_(1)==1,'AdaBoost perfect stop')
  call require(all(model%predict(x)==4),'AdaBoost constant targets')
  call require(all(model%estimator_errors_(2:)==1),'AdaBoost unused error slots')
  model%max_depth=1; model%loss='linear'; y=0; y(20)=10
  call model%fit(0*x,y)
  call require(model%n_estimators_>=1,'AdaBoost retained first estimator')
  print *, 'AdaBoostRegressor tests passed'
end program
