program reference_gpc_advanced_driver
  use sklearn_gp_classification, only: gaussian_process_classifier
  use sklearn_gp_kernels
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  implicit none
  type(gaussian_process_classifier) :: model
  type(gp_kernel) :: kernel,copy
  real(dp), allocatable :: x(:,:),q(:,:),bounds(:,:),theta(:,:),packed(:),gradient(:),before(:,:),values(:)
  integer, allocatable :: labels(:)
  integer :: code,n,p,m,config,seed,opt,restarts,warm,limit,nb,i,d,heads
  real(dp) :: value,bad
  character(80) :: argument
  if(command_argument_count()>0) then
    call get_command_argument(1,argument);call reject(trim(argument));stop 2
  end if
  read(*,*) code,n,p,m,config,seed,opt,restarts,warm,limit,nb
  allocate(x(n,p),q(m,p),labels(n));read(*,*) x;read(*,*) q;read(*,*) labels
  kernel=make_kernel(code,p)
  if(nb>=0) then
    allocate(bounds(nb,2));if(nb>0) read(*,*) bounds
    kernel=kernel%with_bounds(bounds)
  end if
  model%kernel=kernel;model%random_state=seed;model%n_restarts_optimizer=restarts
  model%warm_start=warm/=0;model%max_iter_predict=limit
  if(opt/=0) model%optimizer='fmin_l_bfgs_b'
  call model%fit(x,labels);heads=size(model%kernels_);d=kernel%n_dims();allocate(theta(d,heads))
  do i=1,heads
    theta(:,i)=model%kernels_(i)%theta()
  end do
  call matrix(model%predict_proba(q));call vector(real(model%predict(q),dp))
  call vector([model%log_marginal_likelihood()]);call vector(model%log_marginal_likelihoods_)
  call matrix(model%pi_);call matrix(model%w_sqrt_);call tensor(model%l_);call matrix(model%alpha_)
  call matrix(model%posterior_mode_);call matrix(model%mode_cache_);call matrix(theta)
  call matrix(model%fit_initial_cache_)
  call tensor(model%optimization_points_);call matrix(model%optimization_values_)
  call matrix(real(merge(1,0,model%optimization_run_converged_),dp))
  call tensor(model%optimization_eval_cache_)
  call tensor(model%optimization_initial_points_)
  call vector(real([model%n_optimization_runs_,model%n_classes_,heads],dp))
  before=model%predict_proba(q)
  if(heads==1) then
    value=model%log_marginal_likelihood(theta(:,1),gradient)
    call vector([value]);call vector(gradient);call matrix(model%mode_cache_)
    value=model%log_marginal_likelihood(theta(:,1)+.04_dp,gradient)
    call vector([value]);call vector(gradient);call matrix(model%mode_cache_)
  else
    value=model%log_marginal_likelihood(theta(:,1));call vector([value]);call matrix(model%mode_cache_)
    packed=reshape(theta,[size(theta)])+.04_dp
    value=model%log_marginal_likelihood(packed);call vector([value]);call matrix(model%mode_cache_)
  end if
  do i=1,heads
    call vector(model%kernels_(i)%theta())
  end do
  call matrix(model%predict_proba(q))
  packed=reshape(theta,[size(theta)])+.02_dp
  value=model%log_marginal_likelihood(packed,clone_kernel=.false.);call vector([value]);call matrix(model%mode_cache_)
  do i=1,heads
    call vector(model%kernels_(i)%theta())
  end do
  call matrix(model%predict_proba(q))
  call vector([model%log_marginal_likelihood()])
contains
  function make_kernel(code,p) result(k)
    integer, intent(in) :: code,p
    type(gp_kernel) :: k
    real(dp) :: lengths(p)
    integer :: j
    lengths=[(.6_dp+.4_dp*real(j-1,dp)/max(1,p-1),j=1,p)]
    select case(code)
    case(0);k=1.2_dp*rbf_covariance(.8_dp)+white_covariance(.07_dp)
    case(1);k=matern_covariance(lengths,nu=1.5_dp)+white_covariance(.05_dp)
    case(2);k=rational_quadratic_covariance(.9_dp,.7_dp)+white_covariance(.04_dp)
    case(3);k=periodic_covariance(.9_dp,1.8_dp)+white_covariance(.04_dp)
    case(4);k=dot_product_covariance(.6_dp)+white_covariance(.07_dp)
    case(5);k=constant_covariance(1.2_dp,.true.)*rbf_covariance(.8_dp,.true.)+white_covariance(.04_dp,.true.)
    case(6);k=rbf_covariance(lengths)+white_covariance(.05_dp)
    case(7);k=matern_covariance(lengths,nu=.7_dp)+white_covariance(.05_dp)
    case(8);k=rbf_covariance(.8_dp)*periodic_covariance(.9_dp,1.8_dp)+white_covariance(.04_dp)
    case default;stop 2
    end select
  end function
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *,shape(a);print '(*(es26.17e3,1x))',reshape(a,[size(a)])
  end subroutine
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    print *,size(a);print '(*(es26.17e3,1x))',a
  end subroutine
  subroutine tensor(a)
    real(dp), intent(in) :: a(:,:,:)
    print *,shape(a);print '(*(es26.17e3,1x))',reshape(a,[size(a)])
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: data(6,1)
    integer :: y(6)
    data(:,1)=[-1._dp,-.6_dp,-.2_dp,.2_dp,.6_dp,1._dp];y=[-4,-4,-4,9,9,9]
    bad=ieee_value(0._dp,ieee_quiet_nan);model%kernel=make_kernel(0,1)
    select case(name)
    case('optimizer');model%optimizer='unsupported'
    case('restart');model%n_restarts_optimizer=-1
    case('seed');model%random_state=-1
    case('iterations');model%max_iter=0
    case('evaluations');model%max_fun=0
    case('tolerance');model%tol=-1
    case('nan_tolerance');model%ftol=bad
    case('restart_infinite')
      kernel=rbf_covariance();model%kernel=kernel%with_bounds(reshape([ieee_value(0._dp,ieee_negative_inf), &
        1._dp],[1,2]));model%optimizer='fmin_l_bfgs_b';model%n_restarts_optimizer=1
    case('unfitted_theta');value=model%log_marginal_likelihood([0._dp]);return
    case('cached_gradient');call model%fit(data,y);value=model%log_marginal_likelihood(eval_gradient=.true.);return
    case('theta_shape');call model%fit(data,y);value=model%log_marginal_likelihood([0._dp]);return
    case('multiclass_gradient')
      y=[-4,-4,9,9,17,17];call model%fit(data,y)
      value=model%log_marginal_likelihood(model%kernels_(1)%theta(),gradient);return
    case('multiclass_shape')
      y=[-4,-4,9,9,17,17];call model%fit(data,y);value=model%log_marginal_likelihood([0._dp]);return
    case default;stop 2
    end select
    call model%fit(data,y)
  end subroutine
end program
