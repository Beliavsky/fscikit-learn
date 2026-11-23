program test_positive_linear
  use sklearn_common, only: dp,require
  use sklearn_linear_model, only: linear_regression,ridge,elastic_net,lasso,elastic_net_cv
  use sklearn_weighted_linear, only: weighted_linear
  implicit none
  type(linear_regression) :: ols
  type(ridge) :: rr
  type(elastic_net) :: en
  type(lasso) :: ls
  type(elastic_net_cv) :: cv
  real(dp) :: x(8,3),y(8),weights(8),wide_x(5,9),wide_y(5),gradient(9)
  real(dp), allocatable :: large_x(:,:),large_y(:)
  x(:,1)=[-1._dp,1._dp,-1._dp,1._dp,-1._dp,1._dp,-1._dp,1._dp]
  x(:,2)=[-1._dp,-1._dp,1._dp,1._dp,-1._dp,-1._dp,1._dp,1._dp]
  x(:,3)=[-1._dp,-1._dp,-1._dp,-1._dp,1._dp,1._dp,1._dp,1._dp]
  y=4-2*x(:,1)+3*x(:,2)-x(:,3);weights=1
  ols%positive=.true.;call ols%fit(x,y)
  call require(maxval(abs(ols%coef_-[0._dp,3._dp,0._dp]))<1.e-12_dp,'positive OLS active constraint')
  call require(abs(ols%intercept_-4)<1.e-12_dp.and.ols%rank_==0,'positive OLS intercept and rank sentinel')
  call weighted_linear(ols,x,y,weights,0._dp)
  call require(maxval(abs(ols%coef_-[0._dp,3._dp,0._dp]))<1.e-12_dp,'weighted positive OLS')
  rr%positive=.true.;rr%alpha=8;call rr%fit(x,y)
  call require(maxval(abs(rr%coef_-[0._dp,1.5_dp,0._dp]))<1.e-12_dp,'positive Ridge augmented NNLS')
  en%positive=.true.;en%alpha=.5_dp;en%l1_ratio=.4_dp;call en%fit(x,y)
  call require(abs(en%coef_(2)-2.8_dp/1.3_dp)<1.e-12_dp.and.minval(en%coef_)>=0,'positive elastic coordinate')
  ls%positive=.true.;ls%alpha=.5_dp;call ls%fit(x,y)
  call require(abs(ls%coef_(2)-2.5_dp)<1.e-12_dp.and.minval(ls%coef_)>=0,'positive Lasso')
  cv%positive=.true.;cv%alphas=[1._dp,.1_dp,.01_dp];cv%l1_ratios=[.5_dp,1._dp];cv%cv=4
  call cv%fit(x,y)
  call require(minval(cv%coef_)>=0.and.cv%all_cv_converged_,'positive coordinate CV refit')
  ols%positive=.false.;call ols%fit(x,y)
  call require(maxval(abs(ols%coef_-[-2._dp,3._dp,-1._dp]))<1.e-12_dp,'unconstrained refit restores signed solution')
  wide_x=reshape([ &
    1.76405234596766403e+00_dp,4.10598501938372329e-01_dp,3.13067701650901364e-01_dp, &
    -1.87183850025833598e-01_dp,1.23029068072772074e+00_dp,4.00157208367223294e-01_dp, &
    1.44043571160877987e-01_dp,-8.54095739301724777e-01_dp,1.53277921435845754e+00_dp, &
    1.20237984878441129e+00_dp,9.78737984105739200e-01_dp,1.45427350696297508e+00_dp, &
    -2.55298981583407869e+00_dp,1.46935876990028502e+00_dp,-3.87326817407952273e-01_dp, &
    2.24089319920145780e+00_dp,7.61037725146993416e-01_dp,6.53618595440360606e-01_dp, &
    1.54947425696916302e-01_dp,-3.02302750575335566e-01_dp,1.86755799014996748e+00_dp, &
    1.21675016492828414e-01_dp,8.64436198859505733e-01_dp,3.78162519602173564e-01_dp, &
    -1.04855296506709261e+00_dp,-9.77277879876411015e-01_dp,4.43863232745425662e-01_dp, &
    -7.42165020406441944e-01_dp,-8.87785747630112754e-01_dp,-1.42001793717897518e+00_dp, &
    9.50088417525589368e-01_dp,3.33674327374266833e-01_dp,2.26975462398760763e+00_dp, &
    -1.98079646822392696e+00_dp,-1.70627019062501262e+00_dp,-1.51357208297697887e-01_dp, &
    1.49407907315760613e+00_dp,-1.45436567459876476e+00_dp,-3.47912149326152609e-01_dp, &
    1.95077539523178967e+00_dp,-1.03218851793557845e-01_dp,-2.05158263765800875e-01_dp, &
    4.57585173014460680e-02_dp,1.56348969103980051e-01_dp,-5.09652181751653499e-01_dp &
    ],[5,9])
  wide_y=[ &
    7.49721214882521636e+00_dp,1.57684661213745247e-01_dp,8.48187520856957100e+00_dp, &
    9.78008349436698232e-01_dp,-2.39644896850793199e+00_dp]
  ols%positive=.true.;ols%fit_intercept=.false.;call ols%fit(wide_x,wide_y)
  gradient=matmul(transpose(wide_x),ols%predict(wide_x)-wide_y)
  call require(minval(gradient)>=-1.e-11_dp,'NNLS corrected QR tail satisfies inactive KKT')
  call require(maxval(abs(gradient*ols%coef_))<1.e-10_dp,'NNLS corrected QR tail complementarity')
  allocate(large_x(2,10000),large_y(2));large_x=0;large_x(1,1)=1;large_x(2,5000)=1;large_y=[2._dp,3._dp]
  call ols%fit(large_x,large_y)
  call require(count(ols%coef_>0)==2.and.ols%coef_(1)==2.and.ols%coef_(5000)==3,'wide positive OLS without Gram storage')
  print *, 'Positive linear-model constraints: PASS'
end program
