program test_glm_controls
  use sklearn
  use sklearn_common,only:require
  use sklearn_parameter_space,only:parameter_set,param
  use sklearn_base_dispatch,only:base_weighted,train_regressor
  use,intrinsic::ieee_arithmetic,only:ieee_is_nan
  use sklearn_sgd_clone,only:reset_sgd_fit
  use sklearn_search_adapters,only:set_search_params
  implicit none
  type(tweedie_regressor)::a,b
  type(poisson_regressor)::poisson
  type(gamma_regressor)::gamma
  type(pipeline)::pipe
  real(dp)::x(12,2),y(12),weight(12),before(2),saved_score
  type(parameter_set)::parameters
  integer::i
  a%solver='native-newton';poisson%solver='native-newton';gamma%solver='native-newton'
  do i=1,12
    x(i,:)=[real(i-6,dp)/12,real(mod(i,3),dp)/6]
  end do
  y=exp(0.4_dp+0.3_dp*x(:,1)-0.2_dp*x(:,2));weight=1;weight(:3)=0
  a%power=1.5_dp;a%alpha=0.2_dp;a%tol=1.e-11_dp
  call a%fit(x,y,weight)
  b=a;call b%fit(x(4:,:),y(4:))
  call require(maxval(abs(a%coef_-b%coef_))<1.e-9_dp,'GLM zero weights remove rows')
  call b%fit(x,reshape(y,[12,1]),10*weight)
  call require(maxval(abs(a%coef_-b%coef_))<1.e-9_dp,'GLM normalized weights and column targets')
  call require(abs(a%score(x,y,weight)-b%score(x,y,weight))<1.e-10_dp,'weighted GLM D2')
  saved_score=a%score(x,y,weight);a%power=3
  call require(a%score(x,y,weight)==saved_score,'GLM scoring uses fitted loss')
  a%power=1.5_dp
  a%warm_start=.true.;a%max_iter=1;before=a%coef_;call a%fit(x,y,weight)
  call require(all(a%coef_==before).and.a%n_iter_==0,'GLM warm initial point')
  call reset_sgd_fit(a)
  call require(.not.allocated(a%coef_).and.a%warm_start.and.a%power==1.5_dp,'GLM clone clears state')
  call require(base_weighted(poisson).and.base_weighted(gamma),'GLM weighted composite support')
  call train_regressor(poisson,x,y,weight);call gamma%fit(x,y,2._dp)
  call require(poisson%power_==1.and.gamma%power_==2,'GLM derived weighted dispatch')
  allocate(pipe%steps(1));pipe%steps(1)%name='glm'
  allocate(pipe%steps(1)%model,source=poisson);call pipe%fit(x,y,weight)
  call require(maxval(abs(pipe%predict(x)-poisson%predict(x)))<1.e-10_dp,'weighted GLM Pipeline fit')
  call require(abs(pipe%score(x,y,weight)-poisson%score(x,y,weight))<1.e-10_dp,'weighted GLM Pipeline D2')
  allocate(parameters%items(2))
  parameters%items(1)%name='power';parameters%items(1)%value=param(-1._dp)
  parameters%items(2)%name='link';parameters%items(2)%value=param('log')
  call set_search_params(a,parameters)
  call require(a%power==-1.and.a%link=='log','GLM search controls')
  b%power=0;b%warm_start=.false.;call b%fit(x,spread(1._dp,1,12))
  call require(ieee_is_nan(b%score(x,spread(1._dp,1,12))),'GLM undefined constant D2')
  print *,'weighted GLM controls: PASS'
end program
