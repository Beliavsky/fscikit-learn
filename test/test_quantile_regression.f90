program test_quantile_regression
  use sklearn, only: dp,quantile_regressor
  use sklearn_common, only: require
  implicit none
  type(quantile_regressor) :: model
  real(dp) :: x(9,2),y(9),w(9),pred(9)
  integer :: i
  do i=1,9
    x(i,:)=[real(i-5,dp),sin(real(i,dp))]
  end do
  y=2+3*x(:,1)-.5_dp*x(:,2); model%alpha=0
  call model%fit(x,y)
  call require(model%converged_,'quantile exact-line convergence')
  call require(maxval(abs(model%coef_-[3._dp,-.5_dp]))<1.e-10_dp,'quantile exact slopes')
  call require(abs(model%intercept_-2)<1.e-10_dp,'quantile exact intercept')
  call require(model%score(x,y)>1-1.e-12_dp,'quantile exact R2')
  call require(model%objective_<1.e-10_dp.and.model%dual_gap_<1.e-10_dp,'quantile exact certificate')
  y=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp,7._dp,8._dp,1000._dp]
  w=1; w(9)=0; w(1)=2; model%alpha=10; model%quantile=.5_dp
  call model%fit(x,y,w)
  call require(maxval(abs(model%coef_))<1.e-12_dp,'quantile high penalty zero slopes')
  call require(abs(model%intercept_-4)<1.e-10_dp,'weighted median intercept')
  call require(model%dual_coef_(9)==0,'zero-weight dual entry')
  call require(abs(sum(model%dual_coef_))<1.e-10_dp,'unpenalized intercept dual constraint')
  call require(model%dual_gap_<1.e-10_dp,'weighted quantile dual gap')
  model%max_iter=1; model%alpha=0; model%fit_intercept=.false.
  call model%fit(x,y)
  call require(.not.model%converged_.and.model%n_iter_==1,'quantile iteration limit diagnostics')
  pred=model%predict(x)
  model%max_iter=10000; call model%fit(x,y)
  call require(model%converged_.and.model%intercept_==0,'quantile refit without intercept')
  print *, 'QuantileRegressor tests passed'
end program
