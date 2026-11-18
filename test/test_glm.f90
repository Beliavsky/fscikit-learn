program test_glm
  use sklearn
  use sklearn_common, only: require,normal_random,seed_rng
  implicit none
  type(poisson_regressor) :: poisson
  type(gamma_regressor) :: gamma
  type(tweedie_regressor) :: tweedie
  type(ridge) :: rr
  real(dp) :: x(80,3),y(80),coef(3)
  real(dp), allocatable :: scores(:)
  call seed_rng(31); x=normal_random(80,3)*0.3_dp; coef=[0.4_dp,-0.2_dp,0.1_dp]
  y=exp(matmul(x,coef)+0.3_dp)
  poisson%alpha=0; poisson%tol=1.e-10_dp; call poisson%fit(x,y)
  call require(poisson%converged_.and.maxval(abs(poisson%coef_-coef))<1.e-8_dp,'Poisson log link recovery')
  call require(poisson%score(x,y)>0.999999_dp,'Poisson D2 score')
  gamma%alpha=0; gamma%tol=1.e-10_dp; call gamma%fit(x,y)
  call require(gamma%converged_.and.maxval(abs(gamma%coef_-coef))<1.e-8_dp,'Gamma log link recovery')
  tweedie%power=1.5_dp; tweedie%alpha=0; tweedie%tol=1.e-10_dp; call tweedie%fit(x,y)
  call require(tweedie%converged_.and.maxval(abs(tweedie%coef_-coef))<1.e-8_dp,'compound Tweedie recovery')
  tweedie%power=1+1.e-8_dp; call tweedie%fit(x,y)
  call require(maxval(abs(tweedie%coef_-coef))<1.e-8_dp,'Tweedie stable near Poisson power')
  call require(tweedie%score(x,y)>0.999999_dp,'Tweedie deviance near Poisson power')
  tweedie%power=2-1.e-8_dp; call tweedie%fit(x,y)
  call require(maxval(abs(tweedie%coef_-coef))<1.e-8_dp,'Tweedie stable near Gamma power')
  call require(tweedie%score(x,y)>0.999999_dp,'Tweedie deviance near Gamma power')
  scores=cross_val_score(poisson,x,y,cv=4)
  call require(minval(scores)>0.999999_dp,'GLM cross-validation uses inverse link and D2')
  tweedie%power=0; tweedie%alpha=0.7_dp; call tweedie%fit(x,y)
  rr%alpha=80*tweedie%alpha; call rr%fit(x,y)
  call require(maxval(abs(tweedie%predict(x)-rr%predict(x)))<1.e-10_dp,'power zero ridge normalization')
  call require(abs(tweedie%score(x,y)-rr%score(x,y))<1.e-10_dp,'power zero D2 equals R2')
  x(:,3)=2*x(:,1); poisson%alpha=0; call poisson%fit(x,y)
  call require(poisson%converged_.and.all(poisson%predict(x)>0),'singular unpenalized GLM')
  y(:10)=0; poisson%alpha=0.1_dp; call poisson%fit(x,y)
  call require(poisson%converged_.and.all(poisson%predict(x)>0),'Poisson supports zero observations')
  gamma%max_iter=1; call gamma%fit(x,y+0.1_dp)
  call require(.not.gamma%converged_.and.gamma%n_iter_==1,'GLM capped fit status')
  poisson%fit_intercept=.false.; call poisson%fit(x,y)
  call require(poisson%intercept_==0,'GLM refit removes intercept')
  print *, 'generalized linear models: PASS'
end program
