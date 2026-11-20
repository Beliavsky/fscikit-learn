program test_lars_selection
  use sklearn, only: dp,lars_cv,lasso_lars_cv,lasso_lars_ic
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(lars_cv) :: lar
  type(lasso_lars_cv) :: sparse
  type(lasso_lars_ic) :: information
  real(dp) :: x(48,5),y(48),wide(12,10),wide_y(12)
  integer :: i,j
  do i=1,48
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),sin(real(2*i,dp)),cos(real(3*i,dp)),sin(real(4*i,dp))]
  end do
  y=2+3*x(:,2)-2*x(:,4)+.2_dp*cos(real([(i,i=1,48)],dp)*.7_dp)
  lar%cv=4; call lar%fit(x,y)
  call require(lar%score(x,y)>.99_dp,'LarsCV fitted score')
  call require(size(lar%mse_path_,2)==4.and.size(lar%mse_path_,1)==size(lar%cv_alphas_),'CV path shape')
  call require(all(lar%cv_alphas_(2:)>=lar%cv_alphas_(:size(lar%cv_alphas_)-1)),'ascending CV grid')
  sparse%cv=4; sparse%max_n_alphas=3; call sparse%fit(x,y)
  call require(sparse%score(x,y)>.98_dp,'LassoLarsCV fitted score')
  call require(any(sparse%cv_alphas_==sparse%alpha_),'CV alpha comes from evaluated grid')
  sparse%max_n_alphas=1; call sparse%fit(x,y)
  call require(size(sparse%cv_alphas_)==1.and.sparse%score(x,y)>.99_dp,'single-alpha CV interpolation')
  sparse%max_n_alphas=3; sparse%positive=.true.; call sparse%fit(x,y)
  call require(all(sparse%coef_>=-1.e-10_dp),'positive LassoLarsCV')
  call information%fit(x,y)
  call require(information%noise_variance_>0.and.information%score(x,y)>.99_dp,'AIC variance and predictions')
  call require(size(information%criterion_)==size(information%alphas_),'IC path shape')
  call require(information%alpha_==information%alphas_(minloc(information%criterion_,dim=1)),'IC minimum selection')
  information%criterion='bic'; information%noise_variance=.04_dp; call information%fit(x,y)
  call require(information%noise_variance_==.04_dp,'provided variance retained')
  information%noise_variance=-1; information%positive=.true.; call information%fit(x,y)
  call require(all(information%coef_>=-1.e-10_dp),'positive IC fit and NNLS noise estimate')
  lar%max_iter=0; call lar%fit(x,y)
  call require(lar%n_iter_==0.and.all(lar%coef_==0),'zero-iteration CV path')
  do i=1,12
    do j=1,10
      wide(i,j)=cos(real(i*(j+1),dp))+.03_dp*sin(real(i+j*j,dp))
    end do
    wide_y(i)=2*wide(i,2)-1.5_dp*wide(i,4)+.1_dp*cos(.9_dp*i)
  end do
  lar=lars_cv(); lar%cv=3; call lar%fit(wide,wide_y)
  call require(all(ieee_is_finite(lar%predict(wide))),'default-epsilon rank-deficient CV predictions')
  call require(lar%alpha_==lar%cv_alphas_(minloc(sum(lar%mse_path_,dim=2),dim=1)),'wide CV minimum selection')
  print *, 'Least-angle model selection tests passed'
end program
