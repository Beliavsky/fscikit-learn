program robust_regression
  use sklearn
  implicit none
  type(linear_regression) :: ols
  type(huber_regressor) :: huber
  type(ransac_regressor) :: ransac
  real(dp) :: x(100,2),clean(100),contaminated(100)
  integer :: i
  do i=1,100
    x(i,:)=[sin(real(i,dp)),cos(0.7_dp*i)]
    clean(i)=2*x(i,1)-3*x(i,2)+1
    contaminated(i)=clean(i)+0.04_dp*sin(17._dp*i)
  end do
  contaminated(:15)=contaminated(:15)+12
  call ols%fit(x,contaminated)
  huber%max_iter=500; call huber%fit(x,contaminated)
  ransac%residual_threshold=0.2_dp; ransac%random_state=21
  call ransac%fit(x,contaminated)
  print *, 'R2 against the known clean signal after fitting contaminated targets:'
  print *, 'Ordinary least squares:',ols%score(x,clean)
  print *, 'Huber:',huber%score(x,clean),' converged:',huber%converged_
  print *, 'RANSAC:',ransac%score(x,clean),' inliers:',count(ransac%inlier_mask_)
end program
