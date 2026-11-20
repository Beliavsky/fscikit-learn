program test_least_angle
  use sklearn, only: dp,lars,lasso_lars,lars_path,lars_path_result,linear_regression,lasso
  use sklearn_common, only: require
  implicit none
  type(linear_regression) :: ols
  type(lasso) :: coordinate
  type(lars) :: model
  type(lasso_lars) :: sparse
  type(lars_path_result) :: path
  real(dp) :: x(40,4),y(40)
  real(dp), allocatable :: prediction(:)
  integer :: i
  do i=1,40
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),sin(real(2*i,dp)),cos(real(3*i,dp))]
  end do
  y=2+3*x(:,2)-2*x(:,4); call model%fit(x,y)
  call require(maxval(abs(model%coef_-[0._dp,3._dp,0._dp,-2._dp]))<1.e-7_dp,'Lars reaches least squares')
  call require(abs(model%intercept_-2)<1.e-7_dp,'Lars intercept')
  call require(model%score(x,y)>1-1.e-12_dp,'Lars score')
  sparse%alpha=.1_dp; call sparse%fit(x,y)
  coordinate%alpha=.1_dp; coordinate%tol=1.e-12_dp; call coordinate%fit(x,y)
  call require(maxval(abs(sparse%coef_-coordinate%coef_))<1.e-7_dp,'LassoLars agrees with coordinate-descent optimum')
  call require(size(sparse%coef_path_,2)==size(sparse%alphas_),'path shape')
  call require(abs(sparse%alphas_(size(sparse%alphas_))-.1_dp)<1.e-7_dp,'alpha interpolation endpoint')
  call require(all(sparse%alphas_(:size(sparse%alphas_)-1)>=sparse%alphas_(2:)),'descending path alphas')
  sparse%positive=.true.; call sparse%fit(x,y)
  call require(all(sparse%coef_>=-1.e-12_dp),'positive LassoLars')
  sparse%alpha=100; call sparse%fit(x,y); call require(all(sparse%coef_==0),'large alpha selects zero model')
  sparse%fit_path=.false.; call sparse%fit(x,y)
  call require(.not.allocated(sparse%coef_path_).and..not.allocated(sparse%active_),'fit_path false clears paths')
  call require(size(sparse%alphas_)==1,'fit_path false endpoint alpha')
  path=lars_path(x,y,max_iter=0)
  call require(path%n_iter==0.and.size(path%alphas)==1,'zero iteration path')
  path=lars_path(x,y,method='lasso',alpha_min=.01_dp)
  call require(size(path%coefficients,1)==4,'direct uncentered path')
  x(:,4)=x(:,2); call model%fit(x,y)
  prediction=model%predict(x); call ols%fit(x,y)
  call require(maxval(abs(prediction-ols%predict(x)))<1.e-7_dp,'duplicate-feature predictions match least squares')
  path=lars_path(x,[ (0._dp,i=1,40) ])
  call require(path%n_iter==0.and.all(path%coefficients==0),'zero response path')
  print *, 'Lars and LassoLars tests passed'
end program
