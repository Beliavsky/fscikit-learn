program reference_robust_glm_driver
  use sklearn
  implicit none
  type(huber_regressor) :: huber
  type(ransac_regressor) :: ransac
  type(poisson_regressor) :: poisson
  type(gamma_regressor) :: gamma
  type(tweedie_regressor) :: tweedie
  real(dp), allocatable :: x(:,:),yr(:),yp(:),yg(:),yt(:),query(:,:),target(:)
  real(dp) :: powers(5)=[0._dp,1._dp,1.5_dp,1.9_dp,2._dp]
  integer :: n,p,i,input,output
  character(1024) :: path
  character(32) :: name
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),yr(n),yp(n),yg(n),yt(n))
  do i=1,n
    read(input,*) x(i,:),yr(i),yp(i),yg(i),yt(i)
  end do
  close(input); query=x+0.017_dp
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  huber%max_iter=1000; huber%tol=1.e-6_dp; huber%alpha=0.01_dp
  call huber%fit(x,yr); call huber_results('huber')
  huber%epsilon=1.8_dp; huber%alpha=0.2_dp; huber%fit_intercept=.false.
  call huber%fit(x,yr); call huber_results('huber_no_intercept')
  huber%epsilon=2; huber%alpha=0; huber%fit_intercept=.true.
  call huber%fit(x,yr); call huber_results('huber_unpenalized')
  ransac%random_state=21; ransac%stop_probability=0.999999_dp; ransac%max_trials=1000
  ransac%residual_threshold=0.5_dp; call ransac%fit(x,yr); call ransac_results('ransac',yr)
  ransac%loss='squared_error'; ransac%residual_threshold=0.25_dp
  call ransac%fit(x,yr); call ransac_results('ransac_squared',yr)
  ransac%loss='absolute_error'; ransac%residual_threshold=-1
  call ransac%fit(x,yr); call ransac_results('ransac_auto',yr)
  ransac%fit_intercept=.false.; ransac%residual_threshold=0.5_dp
  call ransac%fit(x,yr-1.2_dp); call ransac_results('ransac_no_intercept',yr-1.2_dp)
  poisson%alpha=0.3_dp; poisson%tol=1.e-10_dp; poisson%max_iter=500
  call poisson%fit(x,yp); call glm_results('poisson',poisson,yp)
  poisson%fit_intercept=.false.; poisson%alpha=0
  call poisson%fit(x,yp); call glm_results('poisson_no_intercept',poisson,yp)
  gamma%alpha=0.4_dp; gamma%tol=1.e-10_dp; gamma%max_iter=500
  call gamma%fit(x,yg); call glm_results('gamma',gamma,yg)
  gamma%fit_intercept=.false.; gamma%alpha=0
  call gamma%fit(x,yg); call glm_results('gamma_no_intercept',gamma,yg)
  tweedie%tol=1.e-10_dp; tweedie%max_iter=500; tweedie%alpha=0.2_dp
  do i=1,5
    tweedie%power=powers(i); target=yt
    if(i==1) target=yr
    if(i==2) target=yp
    if(i==5) target=yg
    call tweedie%fit(x,target); write(name,'(a,i0)') 'tweedie_',i
    call glm_results(trim(name),tweedie,target)
  end do
  poisson%max_iter=1; poisson%alpha=0.3_dp; poisson%fit_intercept=.true.
  call poisson%fit(x,yp); call glm_results('poisson_capped',poisson,yp)
  close(output)
contains
  subroutine huber_results(prefix)
    character(*), intent(in) :: prefix
    call vector(prefix//'_coef',huber%coef_)
    call vector(prefix//'_parameters',[huber%intercept_,huber%scale_])
    call vector(prefix//'_predict',huber%predict(query))
    call vector(prefix//'_outliers',merge(1._dp,0._dp,huber%outliers_))
    call vector(prefix//'_loss',[huber%loss_])
  end subroutine
  subroutine ransac_results(prefix,target)
    character(*), intent(in) :: prefix
    real(dp), intent(in) :: target(:)
    call vector(prefix//'_coef',ransac%coef_)
    call vector(prefix//'_parameters',[ransac%intercept_,ransac%score(x,target)])
    call vector(prefix//'_predict',ransac%predict(query))
    call vector(prefix//'_inliers',merge(1._dp,0._dp,ransac%inlier_mask_))
  end subroutine
  subroutine glm_results(prefix,model,target)
    character(*), intent(in) :: prefix
    class(tweedie_regressor), intent(in) :: model
    real(dp), intent(in) :: target(:)
    call vector(prefix//'_coef',model%coef_)
    call vector(prefix//'_parameters',[model%intercept_,merge(1._dp,0._dp,model%converged_)])
    call vector(prefix//'_predict',model%predict(query))
    call vector(prefix//'_score',[model%score(x,target)])
  end subroutine
  subroutine vector(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') name,values
  end subroutine
end program
