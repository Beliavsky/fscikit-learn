program test_robust_regression
  use sklearn, only: dp,huber_regressor,ransac_regressor,linear_regression
  use sklearn_common, only: require,normal_random,seed_rng
  implicit none
  type(huber_regressor) :: huber
  type(ransac_regressor) :: ransac,repeat
  type(linear_regression) :: ols
  real(dp) :: x(100,3),y(100),clean(100),coef(3),scale
  call seed_rng(37); x=normal_random(100,3); coef=[2._dp,-1._dp,0.5_dp]
  clean=matmul(x,coef)+0.7_dp; y=clean+0.03_dp*normal_noise()
  y(:15)=y(:15)+20
  huber%max_iter=500; call huber%fit(x,y); call ols%fit(x,y)
  call require(huber%converged_,'Huber convergence')
  call require(maxval(abs(huber%coef_-coef))<0.03_dp,'Huber robust coefficient recovery')
  call require(sum((huber%predict(x)-clean)**2)<sum((ols%predict(x)-clean)**2)*0.01_dp, &
    'Huber improves contaminated least squares')
  call require(all(huber%outliers_(:15)).and.huber%scale_>0,'Huber outlier mask and scale')
  scale=huber%scale_; huber%alpha=0; call huber%fit(x,y)
  coef=huber%coef_; scale=huber%scale_; call huber%fit(x,10*y)
  call require(maxval(abs(huber%coef_/10-coef))<1.e-5_dp,'Huber target scale equivariance')
  call require(abs(huber%scale_/10-scale)<1.e-5_dp,'Huber scale parameter equivariance')
  huber%max_iter=1; call huber%fit(x,y)
  call require(.not.huber%converged_.and.huber%n_iter_==1,'Huber capped fit status')
  huber%max_iter=1000; huber%epsilon=1; call huber%fit(x,y)
  call require(maxval(abs(huber%coef_-[2._dp,-1._dp,0.5_dp]))<0.03_dp,'Huber epsilon-one boundary recovery')
  call require(huber%scale_>=10*epsilon(1._dp),'Huber constrained scale lower bound')
  ransac%random_state=9; ransac%residual_threshold=0.2_dp; call ransac%fit(x,y)
  call require(.not.any(ransac%inlier_mask_(:15)).and.all(ransac%inlier_mask_(16:)), &
    'RANSAC known separated consensus')
  call require(maxval(abs(ransac%predict(x)-clean))<0.03_dp,'RANSAC final inlier refit')
  repeat=ransac; call repeat%fit(x,y)
  call require(all(repeat%inlier_mask_.eqv.ransac%inlier_mask_).and.repeat%n_trials_==ransac%n_trials_, &
    'RANSAC seed repeatability')
  ransac%loss='squared_error'; ransac%residual_threshold=0.04_dp; call ransac%fit(x,y)
  call require(count(ransac%inlier_mask_)==85,'RANSAC squared-loss threshold')
  ransac%loss='absolute_error'; ransac%residual_threshold=-1; call ransac%fit(x,y)
  call require(count(ransac%inlier_mask_)==85,'RANSAC automatic MAD threshold')
  ransac%min_samples=100; ransac%residual_threshold=100; call ransac%fit(x,y)
  call require(ransac%n_trials_==1.and.all(ransac%inlier_mask_),'RANSAC dynamic trial cap')
  print *, 'robust regression: PASS'
contains
  function normal_noise() result(noise)
    real(dp) :: noise(100),matrix(100,1)
    matrix=normal_random(100,1); noise=matrix(:,1)
  end function
end program
