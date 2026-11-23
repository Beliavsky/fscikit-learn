program test_sgd_regularization
  use sklearn, only: dp,sgd_regressor,sgd_classifier
  use sklearn_common, only: require
  implicit none
  type(sgd_regressor) :: model,l2,l1
  type(sgd_classifier) :: classifier
  real(dp) :: x(2,1),y(2),zero(12,3),target(12)
  real(dp), allocatable :: saved(:)
  integer :: i
  x(:,1)=[1._dp,2._dp];y=[2._dp,3._dp]
  model%penalty='l1';model%alpha=.5_dp;model%learning_rate='constant'
  model%eta0=.1_dp;model%max_iter=1;model%tol=-1
  call model%fit(x,y)
  call require(abs(model%coef_(1)-.6_dp)<1.e-14_dp,'two analytic truncated-gradient L1 updates')
  call require(abs(model%intercept_-.45_dp)<1.e-14_dp,'L1 leaves intercept unpenalized')
  model%average=1;call model%fit(x,y)
  call require(abs(model%coef_(1)-.35_dp)<1.e-14_dp,'source lazy average and L1 update order')
  model%average=0;model%max_iter=7
  l1=model;l2=model;l2%penalty='l2';call l1%fit(x,y);call l2%fit(x,y)
  model%penalty='elasticnet';model%l1_ratio=0;call model%fit(x,y)
  call require(all(model%coef_==l2%coef_),'elastic-net zero-ratio L2 endpoint')
  model%l1_ratio=1;call model%fit(x,y)
  call require(all(model%coef_==l1%coef_),'elastic-net unit-ratio L1 endpoint')
  zero=0;target=0;model%learning_rate='adaptive';model%max_iter=100
  model%n_iter_no_change=2;model%tol=.001_dp;call model%fit(zero,target)
  call require(model%n_iter_==19.and.model%converged_,'adaptive rate reductions before convergence')
  model%learning_rate='optimal';model%eta0=0;model%tol=-1;model%max_iter=3
  call model%fit(x,y)
  call require(model%t_==7,'optimal schedule accepts zero eta0 and advances sample clock')
  model%shuffle=.true.;model%random_state=42;call model%fit(x,y);saved=model%coef_
  call model%fit(x,y);call require(all(saved==model%coef_),'local source shuffle is repeatable')
  classifier%penalty='elasticnet';classifier%learning_rate='adaptive';classifier%max_iter=10
  classifier%alpha=.05_dp;classifier%eta0=.03_dp;classifier%shuffle=.true.;classifier%random_state=42
  call classifier%fit(reshape([(real(i,dp)/12,i=1,12)],[12,1]),[(mod(i,3),i=1,12)])
  call require(all(shape(classifier%coef_)==[3,1]),'shared penalties and schedules reach multiclass heads')
  print *, 'SGD penalties, schedules and source shuffling: PASS'
end program
