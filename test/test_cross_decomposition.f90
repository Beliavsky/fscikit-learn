program test_cross_decomposition
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(pls_regression) :: reg
  type(pls_canonical) :: canonical
  type(cca) :: correlation
  type(pls_svd) :: svd
  real(dp) :: x(100,4),y(100,2),coefficients(4,2),t,gram(4,4),small(2,2)
  real(dp), allocatable :: z(:,:),u(:,:),prediction(:,:),restored(:,:),cross(:,:),check(:,:)
  integer :: i,j
  do i=1,100
    t=real(i,dp)/100
    x(i,:)=[sin(17*t),cos(23*t),sin(41*t),t]
  end do
  coefficients=reshape([2._dp,-1._dp,0.5_dp,3._dp,-1._dp,4._dp,2._dp,-2._dp],[4,2])
  y=matmul(x,coefficients)+spread([1._dp,-3._dp],1,100)
  reg%n_components=4; reg%tol=1.e-12_dp; z=reg%fit_transform(x,y); prediction=reg%predict(x)
  call require(reg%converged_,'PLS regression converges')
  call require(maxval(abs(prediction-y))<1.e-9_dp,'full PLS recovers multivariate linear model')
  call require(maxval(abs(transpose(reg%coef_)-coefficients))<1.e-9_dp,'PLS coefficient orientation and scaling')
  restored=reg%inverse_transform(z)
  call require(maxval(abs(restored-x))<1.e-10_dp,'full PLS predictor reconstruction')
  gram=matmul(transpose(reg%x_scores_),reg%x_scores_)
  do j=1,4
    gram(j,j)=0
  end do
  call require(maxval(abs(gram))<1.e-10_dp,'PLS deflated predictor scores orthogonal')
  call require(abs(reg%score(x,y)-1)<1.e-12_dp,'PLS uniform-average target R2')
  call require(maxval(abs(z-reg%x_scores_))<1.e-10_dp,'PLS rotations recover deflated predictor scores')
  canonical%tol=1.e-12_dp; z=canonical%fit_transform(x,y); u=canonical%transform_y(y)
  small=matmul(transpose(canonical%y_scores_),canonical%y_scores_)
  call require(abs(small(1,2))<1.e-10_dp,'canonical response scores orthogonal')
  restored=canonical%inverse_transform_y(u)
  call require(maxval(abs(restored-y))<1.e-9_dp,'canonical full response reconstruction')
  correlation%tol=1.e-12_dp; call correlation%fit(x,y)
  do j=1,2
    t=sum(correlation%x_scores_(:,j)*correlation%y_scores_(:,j))/ &
      sqrt(sum(correlation%x_scores_(:,j)**2)*sum(correlation%y_scores_(:,j)**2))
    call require(abs(t-1)<1.e-9_dp,'CCA recovers perfectly shared linear variates')
  end do
  z=svd%fit_transform(x,y); u=svd%transform_y(y)
  cross=matmul(transpose(z),u)
  call require(abs(cross(1,2))+abs(cross(2,1))<1.e-9_dp,'PLSSVD diagonalizes cross covariance')
  check=matmul(transpose(svd%x_weights_),svd%x_weights_)
  call require(abs(check(1,1)-1)+abs(check(2,2)-1)+abs(check(1,2))<1.e-12_dp, &
    'PLSSVD orthonormal weights')
  reg%n_components=2; y=2; call reg%fit(x,y); prediction=reg%predict(x)
  call require(reg%n_components_fitted_==0.and.size(reg%n_iter_)==0,'constant responses stop cleanly')
  call require(maxval(abs(prediction-2))<1.e-12_dp,'constant responses predict target means')
  print *, 'cross decomposition: PASS'
end program
