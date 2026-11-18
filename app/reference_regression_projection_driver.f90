program reference_regression_projection_driver
  use sklearn
  implicit none
  type(bayesian_ridge) :: br
  type(ard_regression) :: ar
  type(orthogonal_matching_pursuit) :: omp
  type(factor_analysis) :: fa
  type(gaussian_random_projection) :: gp
  type(sparse_random_projection) :: sp
  real(dp), allocatable :: x(:,:),y(:),q(:,:),z(:,:)
  integer :: n,p,i,input,output
  character(1024) :: path
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n))
  do i=1,n
    read(input,*) x(i,:),y(i)
  end do
  close(input); q=x+0.031_dp
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  br%tol=1.e-8_dp
  if(n<p) br%fit_intercept=.false.
  call br%fit(x,y); call bayes_results('bayes')
  br%fit_intercept=.false.; br%alpha_init=0.7_dp; br%lambda_init=1.3_dp
  call br%fit(x,y); call bayes_results('bayes_no_intercept')
  br%max_iter=1; call br%fit(x,y); call bayes_results('bayes_capped')
  ar%tol=1.e-8_dp
  if(n<p) ar%tol=1.e-4_dp
  ar%max_iter=1000; call ar%fit(x,y); call ard_results('ard')
  ar%fit_intercept=.false.; ar%threshold_lambda=200
  call ar%fit(x,y); call ard_results('ard_pruned')
  ar%max_iter=1; call ar%fit(x,y); call ard_results('ard_capped')
  omp%n_nonzero_coefs=2; call omp%fit(x,y)
  call vector('omp_coef',omp%coef_); call vector('omp_predict',omp%predict(q))
  call vector('omp_iter',[real(omp%n_iter_,dp)])
  omp%tol=0.5_dp; call omp%fit(x,y)
  call vector('omp_tol_coef',omp%coef_); call vector('omp_tol_predict',omp%predict(q))
  omp%fit_intercept=.false.; omp%tol=-1; call omp%fit(x,y)
  call vector('omp_no_intercept',omp%predict(q))
  fa%n_components=2; fa%tol=1.e-7_dp; fa%max_iter=3000; call fa%fit(x)
  call factor_results('factor')
  fa%n_components=0; call fa%fit(x); call factor_results('factor_zero')
  fa%n_components=1; fa%max_iter=1; fa%noise_variance_init=spread(0.8_dp,1,p)
  call fa%fit(x); call factor_results('factor_capped')
  gp%n_components=3; gp%compute_inverse_components=.true.; z=gp%fit_transform(x)
  call matrix('gaussian_components',gp%components_); call matrix('gaussian_transform',z)
  call matrix('gaussian_inverse',gp%inverse_transform(z))
  call matrix('gaussian_pinv',gp%inverse_components_)
  sp%n_components=3; z=sp%fit_transform(x)
  call matrix('sparse_components',sp%components_); call matrix('sparse_transform',z)
  call matrix('sparse_inverse',sp%inverse_transform(z))
  sp%density=1; sp%n_components=p+2; sp%compute_inverse_components=.true.; z=sp%fit_transform(x)
  call matrix('dense_sign_components',sp%components_); call matrix('dense_sign_transform',z)
  call matrix('dense_sign_inverse',sp%inverse_transform(z))
  call vector('jl_bound',[real(johnson_lindenstrauss_min_dim(n,0.4_dp),dp)])
  close(output)
contains
  subroutine bayes_results(prefix)
    character(*), intent(in) :: prefix
    call vector(prefix//'_coef',br%coef_); call matrix(prefix//'_sigma',br%sigma_)
    call vector(prefix//'_predict',br%predict(q)); call vector(prefix//'_std',br%predict_std(q))
    call vector(prefix//'_parameters',[br%alpha_,br%lambda_,br%intercept_,real(br%n_iter_,dp)])
  end subroutine
  subroutine ard_results(prefix)
    character(*), intent(in) :: prefix
    call vector(prefix//'_coef',ar%coef_); call matrix(prefix//'_sigma',ar%sigma_)
    call vector(prefix//'_predict',ar%predict(q)); call vector(prefix//'_std',ar%predict_std(q))
    call vector(prefix//'_lambda',ar%lambda_)
    call vector(prefix//'_parameters',[ar%alpha_,ar%intercept_,real(ar%n_iter_,dp)])
  end subroutine
  subroutine factor_results(prefix)
    character(*), intent(in) :: prefix
    real(dp), allocatable :: zz(:,:)
    zz=fa%transform(q)
    call matrix(prefix//'_cov',fa%get_covariance()); call matrix(prefix//'_precision',fa%get_precision())
    call matrix(prefix//'_latent_gram',matmul(zz,transpose(zz)))
    call vector(prefix//'_noise',fa%noise_variance_); call vector(prefix//'_score',fa%score_samples(q))
    call vector(prefix//'_loglike',fa%loglike_); call vector(prefix//'_iter',[real(fa%n_iter_,dp)])
  end subroutine
  subroutine vector(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') name,values
  end subroutine
  subroutine matrix(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:,:)
    call vector(name,reshape(values,[size(values)]))
  end subroutine
end program
