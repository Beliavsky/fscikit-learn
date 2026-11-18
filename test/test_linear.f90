program test_linear
  use sklearn_common
  use sklearn_linear_model
  implicit none
  type(linear_regression) :: ols
  type(ridge) :: rr
  type(lasso) :: l1
  type(elastic_net) :: en
  type(logistic_regression) :: lr
  type(perceptron) :: pc
  real(dp) :: x(6,2),y(6),binary(6,1)
  real(dp), allocatable :: prob(:,:)
  integer :: labels(6),i
  x(:,1)=[-2._dp,-1._dp,0._dp,1._dp,2._dp,3._dp]
  x(:,2)=[1._dp,0._dp,1._dp,0._dp,1._dp,0._dp]
  y=2*x(:,1)-3*x(:,2)+4
  call ols%fit(x,y)
  call require(maxval(abs(ols%coef_-[2._dp,-3._dp]))<1.e-10_dp,'OLS coefficients')
  call require(abs(ols%intercept_-4)<1.e-10_dp,'OLS intercept')
  call require(ols%score(x,y)>0.999999_dp,'OLS score')
  rr%alpha=0; call rr%fit(x,y)
  call require(maxval(abs(rr%predict(x)-y))<1.e-10_dp,'ridge alpha zero')
  l1%alpha=0.01_dp; call l1%fit(x,y)
  call require(l1%converged_.and.l1%score(x,y)>0.999_dp,'lasso convergence')
  en%alpha=0.01_dp; call en%fit(x,y)
  call require(en%converged_.and.en%score(x,y)>0.999_dp,'elastic net')
  binary(:,1)=x(:,1); labels=[-1,-1,-1,2,2,2]
  call lr%fit(binary,labels); prob=lr%predict_proba(binary)
  call require(lr%converged_,'binary logistic convergence')
  call require(all(lr%predict(binary)==labels),'binary logistic labels')
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-12_dp,'logistic probability sum')
  call pc%fit(binary,labels); call require(pc%score(binary,labels)==1,'perceptron')
  labels=[0,0,1,1,2,2]; call lr%fit(x,labels)
  call require(lr%converged_,'multiclass logistic convergence')
  call require(lr%score(x,labels)==1,'multiclass logistic labels')
  ! Rank deficient least squares returns a minimum norm solution.
  x(:,2)=2*x(:,1); y=3*x(:,1)+2; call ols%fit(x,y)
  call require(ols%rank_==1,'OLS rank detection')
  call require(maxval(abs(ols%coef_-[0.6_dp,1.2_dp]))<1.e-10_dp,'OLS minimum norm')
  ! A small but identifiable singular value must survive without A^T A loss.
  x(:,2)=x(:,1)+1.e-7_dp*[1._dp,-1._dp,0._dp,2._dp,-2._dp,1._dp]
  y=2*x(:,1)-3*x(:,2)+4; call ols%fit(x,y)
  call require(maxval(abs(ols%coef_-[2._dp,-3._dp]))<1.e-7_dp,'ill-conditioned OLS')
  print *, 'linear models: PASS'
end program
