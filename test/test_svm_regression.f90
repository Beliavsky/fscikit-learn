program test_svm_regression
  use sklearn
  use sklearn_common, only: require
  use,intrinsic::iso_fortran_env,only:real32
  implicit none
  type(svr) :: model,weighted,removed
  type(nu_svr) :: nu_model
  real(dp) :: x(40,2),y(40),t,weights(40),beta(40),residual(40),eps,cache_error
  real(dp), allocatable :: predicted(:),again(:),gram(:,:),features(:,:)
  integer :: i
  do i=1,40
    t=real(i,dp)/10; x(i,:)=[sin(t),cos(1.7_dp*t)]
    y(i)=sin(2*t)+0.12_dp*sin(17*t)
  end do
  model%c=2; model%gamma=0.7_dp; model%epsilon=0.08_dp; model%tol=1.e-7_dp
  call model%fit(x,y); predicted=model%predict(x)
  call require(model%converged_,'SVR cached-gradient convergence')
  call require(model%score(x,y)>0.9_dp,'SVR nonlinear regression accuracy')
  beta=0; beta(model%support_)=model%dual_coef_; residual=predicted-y
  call require(abs(sum(beta))<1.e-10_dp.and.all(abs(beta)<=model%c+1.e-12_dp),'SVR dual feasibility')
  eps=model%epsilon
  do i=1,40
    if(abs(beta(i))<1.e-10_dp) then
      call require(abs(residual(i))<=eps+2.e-6_dp,'SVR zero dual within epsilon tube')
    else if(abs(beta(i))<model%c-1.e-9_dp) then
      call require(abs(residual(i)+sign(eps,beta(i)))<2.e-6_dp,'SVR free dual on epsilon tube')
    else
      call require(sign(1._dp,beta(i))*residual(i)<=-eps+2.e-6_dp,'SVR bounded dual complementary slackness')
    end if
  end do
  gram=rbf_kernel(x,x,0.7_dp)
  cache_error=maxval(matmul(abs(gram-real(real(gram,real32),dp)),abs(beta)))
  call require(model%kkt_violation_<=model%tol+2*cache_error+1.e-9_dp,'SVR physical residual respects cache rounding bound')
  weighted=model; weighted%kernel='precomputed'
  call weighted%fit(gram,y); again=weighted%predict(gram)
  call require(maxval(abs(predicted-again))<1.e-9_dp,'SVR precomputed kernel equivalence')
  weights=1; weights(40)=0; weighted=model; removed=model
  y(40)=1.e6_dp; call weighted%fit(x,y,weights); call removed%fit(x(:39,:),y(:39))
  call require(maxval(abs(weighted%predict(x)-removed%predict(x)))<1.e-8_dp,'SVR zero weights remove observations')
  call require(weighted%score(x,y,weights)>0.9_dp,'weighted SVR score ignores excluded target')
  y(40)=sin(8._dp)+0.12_dp*sin(68._dp)
  nu_model%c=2; nu_model%nu=0.4_dp; nu_model%gamma=0.7_dp; nu_model%tol=1.e-7_dp
  call nu_model%fit(x,y); predicted=nu_model%predict(x)
  call require(nu_model%converged_,'NuSVR KKT convergence')
  model%epsilon=nu_model%epsilon_; call model%fit(x,y)
  call require(maxval(abs(predicted-model%predict(x)))<1.e-4_dp,'NuSVR and learned-epsilon SVR equivalence')
  beta=0; beta(nu_model%support_)=nu_model%dual_coef_
  call require(abs(sum(beta))<1.e-10_dp.and.sum(abs(beta))<=nu_model%c*nu_model%nu*40+1.e-8_dp, &
    'NuSVR dual mass constraint')
  model%epsilon=100; call model%fit(x,y)
  call require(model%n_support_==0,'large epsilon admits no support vectors')
  again=model%predict(x); call require(maxval(again)-minval(again)==0,'empty-support model predicts a constant')
  y=3; model%epsilon=0; model%gamma=-1; features=0*x; call model%fit(features,y)
  call require(model%gamma_==1.and.maxval(abs(model%predict(features)-3))<1.e-12_dp, &
    'constant-data gamma scale fallback and constant prediction')
  model%epsilon=0.1_dp; model%max_iter=1; call model%fit(x,[(sin(real(i,dp)),i=1,40)])
  call require(.not.model%converged_.and.model%fit_status_==1,'capped SVR exposes nonconvergence')
  print *, 'support vector regression: PASS'
end program
