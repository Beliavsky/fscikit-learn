program test_glm_newton
  use sklearn
  use sklearn_common,only:require
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use sklearn_glm_lbfgs,only:glm_lbfgs_solve
  use sklearn_search_adapters,only:reset_search_clone,set_search_params
  use sklearn_parameter_space,only:parameter_set,param
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  type(poisson_regressor)::model,copy
  type(gamma_regressor)::gamma
  type(tweedie_regressor)::zero
  type(csr_matrix)::sparse,invalid_trial
  type(parameter_set)::parameters
  real(dp)::x(25,3),y(25),weight(25),coef(3),point(1)
  integer::i,iterations
  logical::converged
  character(60)::message
  do i=1,25
    x(i,:)=[real(i-13,dp)/25,sin(real(i,dp))*.2_dp,real(mod(i,4),dp)*.15_dp]
  end do
  coef=[.2_dp,-.3_dp,.4_dp];y=exp(matmul(x,coef)+.3_dp);weight=1;weight(:3)=0
  sparse=csr_from_dense(x,store_zeros=.true.)
  model%solver='newton-cholesky';model%alpha=0;model%tol=1.e-11_dp
  call model%fit(x,y,weight)
  call require(model%converged_.and.maxval(abs(model%coef_-coef))<1.e-8_dp,'Poisson Newton recovery')
  copy=model;call reset_search_clone(copy)
  call require(.not.allocated(copy%coef_).and.copy%solver=='newton-cholesky','Newton solver clone state')
  call copy%fit(sparse,reshape(y,[25,1]),weight)
  call require(maxval(abs(copy%predict(sparse)-model%predict(x)))<1.e-9_dp,'Newton dense/CSR agreement')
  copy%warm_start=.true.;call copy%fit(sparse,y,weight)
  call require(copy%converged_.and.copy%n_iter_==1,'Source Newton checks convergence after a step')
  gamma%solver='newton-cholesky';gamma%tol=1.e-10_dp
  call gamma%fit(sparse,y,2._dp)
  call require(gamma%converged_.and.all(ieee_is_finite(gamma%coef_)),'Gamma scalar-weight CSR Newton')
  allocate(parameters%items(1));parameters%items(1)%name='solver'
  parameters%items(1)%value=param('newton-cholesky');call set_search_params(zero,parameters)
  zero%alpha=0;zero%fit_intercept=.false.;zero%max_iter=1
  call zero%fit(0*x,y)
  call require(zero%n_iter_==0.and.zero%converged_,'Zero-budget fallback initial-gradient convergence')
  call require(index(zero%optimization_message_,'singular/ill-conditioned')>0,'Newton fallback diagnostics')
  invalid_trial%n_rows=1;invalid_trial%n_cols=1;invalid_trial%indptr=[1,4]
  invalid_trial%indices=[1,1,1];invalid_trial%data=[.25_dp,.75_dp,0._dp];point=1000
  call glm_lbfgs_solve(point,[1._dp],[1._dp],1._dp,'log',0._dp,1.e-10_dp,1,iterations, &
    converged,message,sparse=invalid_trial)
  call require(.not.converged.and.iterations==0,'NaN projected gradient must not signal convergence')
  print *,'GLM Newton dense/CSR, cloning, fallback budget and NaN-gradient status: PASS'
end program
