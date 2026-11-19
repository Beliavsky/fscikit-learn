program test_coordinate_cv
  use sklearn, only: dp,elastic_net_cv,lasso_cv,elastic_net
  use sklearn_common, only: require
  implicit none
  type(elastic_net_cv) :: model
  type(lasso_cv) :: sparse
  type(elastic_net) :: fit
  real(dp) :: x(11,2),y(11)
  real(dp), allocatable :: prediction(:)
  integer :: i
  do i=1,11
    x(i,:)=[real(i,dp)/5,real(mod(i,3),dp)]; y(i)=2*x(i,1)-x(i,2)+.7_dp
  end do
  model%alphas=[.1_dp,1._dp,.01_dp]; model%l1_ratios=[.5_dp,1._dp]; model%cv=3; model%tol=1.e-12_dp
  call model%fit(x,y)
  call require(all(model%alphas_(:,1)==[1._dp,.1_dp,.01_dp]),'alpha grid descending')
  call require(all(shape(model%mse_path_)==[3,3,2]),'CV path dimensions')
  fit%alpha=model%alpha_; fit%l1_ratio=model%l1_ratio_; fit%tol=model%tol; call fit%fit(x,y)
  call require(maxval(abs(fit%coef_-model%coef_))<1.e-12_dp,'selected model refitted on full data')
  fit%alpha=1; fit%l1_ratio=.5_dp; call fit%fit(x(5:,:),y(5:))
  prediction=fit%predict(x(:4,:))
  call require(abs(model%mse_path_(1,1,1)-sum((prediction-y(:4))**2)/4)<1.e-12_dp,'first fold excludes test rows')
  call require(abs(model%best_mse_-minval(sum(model%mse_path_,dim=2)/3))<1.e-14_dp,'mean fold MSE selection')
  y=3; call model%fit(x,y)
  call require(model%alpha_==1.and.model%l1_ratio_==.5_dp,'ties choose first ratio and largest alpha')
  call require(all(model%coef_==0).and.abs(model%intercept_-3)<1.e-14_dp,'constant target refit')
  sparse%n_alphas=5; sparse%cv=3; sparse%l1_ratios=[.2_dp]; call sparse%fit(x,y)
  call require(all(sparse%alphas_==1.e-15_dp).and.sparse%l1_ratio_==1,'zero correlation grid and Lasso ratio')
  call require(sparse%converged_.and.sparse%all_cv_converged_,'convergence diagnostics')
  print *, 'Coordinate-descent CV tests passed'
end program
