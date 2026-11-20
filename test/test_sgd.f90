program test_sgd
  use sklearn, only: dp,sgd_regressor
  use sklearn_common, only: require
  implicit none
  type(sgd_regressor) :: model,stream
  real(dp) :: x(2,1),y(2),zero(2,1)
  real(dp), allocatable :: saved(:)
  x(:,1)=[1._dp,2._dp]; y=[2._dp,3._dp]; zero=0
  model%penalty='none'; model%learning_rate='constant'; model%eta0=.1_dp; model%max_iter=1; model%tol=-1
  call model%fit(x,y)
  call require(abs(model%coef_(1)-.68_dp)<1.e-14_dp.and.abs(model%intercept_-.44_dp)<1.e-14_dp,'analytic SGD steps')
  call require(model%t_==3.and.model%n_iter_==1,'sample clock and epoch count')
  stream=model; call stream%fit(x(:1,:),y(:1)); call stream%partial_fit(x(2:,:),y(2:))
  call require(maxval(abs(stream%coef_-model%coef_))<1.e-14_dp,'partial fit preserves coefficient trajectory')
  model%average=1; call model%fit(x,y)
  call require(abs(model%coef_(1)-.44_dp)<1.e-14_dp.and.abs(model%intercept_-.32_dp)<1.e-14_dp,'averaged iterates')
  stream=model; call stream%fit(x(:1,:),y(:1)); call stream%partial_fit(x(2:,:),y(2:))
  call require(maxval(abs(stream%coef_-model%coef_))<1.e-14_dp,'averaging continues across batches')
  model%average=0; model%fit_intercept=.false.; model%penalty='l2'; model%alpha=2
  call model%fit(x,y,sample_weight=[1._dp,0._dp])
  call require(abs(model%coef_(1)-.16_dp)<1.e-14_dp,'zero-weight row still applies L2 decay')
  model%max_iter=20; model%tol=0; model%n_iter_no_change=2; y=0; call model%fit(zero,y)
  call require(.not.model%converged_,'strict no-improvement comparison at zero tolerance')
  model%tol=1.e-3_dp; call model%fit(zero,y)
  call require(model%converged_.and.model%n_iter_==3,'training-loss patience stopping')
  model%shuffle=.true.; model%random_state=42; model%tol=-1; call model%fit(x,[2._dp,3._dp]); saved=model%coef_
  call model%fit(x,[2._dp,3._dp]); call require(all(saved==model%coef_),'native seeded shuffle reproducibility')
  print *, 'SGDRegressor tests passed'
end program
