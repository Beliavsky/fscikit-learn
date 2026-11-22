program reference_gpr_advanced_driver
  use sklearn_gaussian_process, only: gaussian_process_regressor
  use sklearn_gp_kernels
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  implicit none
  type(gaussian_process_regressor) :: model
  type(gp_kernel) :: kernel,copy
  real(dp), allocatable :: x(:,:),q(:,:),y(:,:),weights(:),gradient(:),theta(:),bounds(:,:),variance(:),v(:)
  integer :: mode,code,n,p,m,outputs,config,seed,i,na,nb,opt,restarts
  real(dp) :: value,bad
  character(80) :: argument
  if(command_argument_count()>0) then
    call get_command_argument(1,argument);call reject(trim(argument));stop 2
  end if
  read(*,*) mode,code,n,p,m,outputs,config,seed,opt,restarts,na,nb
  allocate(x(n,p),q(m,p),y(n,outputs),weights(m));read(*,*) x;read(*,*) q;read(*,*) y;read(*,*) weights
  kernel=make_kernel(code,p)
  if(nb>=0) then
    allocate(bounds(nb,2));if(nb>0) read(*,*) bounds
    kernel=kernel%with_bounds(bounds)
  end if
  model%kernel=kernel;model%alpha=.03_dp;model%normalize_y=mod(config,2)==1
  model%random_state=seed;model%n_restarts_optimizer=restarts
  if(opt/=0) model%optimizer='fmin_l_bfgs_b'
  if(na>=0) then
    allocate(model%alpha_values(na));if(na>0) read(*,*) model%alpha_values
  end if
  if(mode==0) then
    model%n_targets=outputs
    call matrix(model%predict_multi(q));call matrix(model%predict_std_multi(q));call tensor(model%predict_cov_multi(q));stop
  end if
  if(mode==1) then
    call model%fit(x,y(:,1))
  else
    call model%fit(x,y)
  end if
  if(mode==3) then
    model%kernel_=rbf_covariance(.8_dp)**(-1)
    value=model%log_marginal_likelihood(model%kernel_%theta(),gradient)
    call vector([value]);call vector(gradient);stop
  end if
  call matrix(model%predict_multi(q));call matrix(model%predict_std_multi(q));call tensor(model%predict_cov_multi(q))
  call vector([model%log_marginal_likelihood_]);call matrix(model%alpha_multi_);call matrix(model%l_)
  call matrix(model%y_train_);call vector(model%y_train_means_);call vector(model%y_train_stds_)
  theta=model%kernel_%theta();call vector(theta);call matrix(model%kernel_%bounds())
  value=model%log_marginal_likelihood(theta,gradient);call vector([value]);call vector(gradient)
  copy=model%kernel_%with_theta(theta+.07_dp)
  value=model%log_marginal_likelihood(theta+.07_dp,gradient);call vector([value]);call vector(gradient)
  call vector(model%kernel_%theta())
  value=model%log_marginal_likelihood(theta+.07_dp,gradient,clone_kernel=.false.)
  call vector([value]);call vector(gradient);call vector(model%kernel_%theta())
  model%kernel_=model%kernel_%with_theta(theta)
  if(mode==1) then
    call vector([model%score(q,y(:m,1),weights)])
    call vector(model%predict(q));call vector(model%predict_std(q));call matrix(model%predict_cov(q))
  else
    call vector([model%score(q,y(:m,:),weights)])
  end if
  call vector(real([model%n_targets_,model%n_features_in_,model%n_optimization_runs_],dp))
  call matrix(model%optimization_points_);call vector(model%optimization_values_)
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
    real(dp) :: data(5,1),target(5,2)
    data(:,1)=[0._dp,.2_dp,.5_dp,.7_dp,1._dp];target(:,1)=sin(data(:,1));target(:,2)=cos(data(:,1))
    bad=ieee_value(0._dp,ieee_quiet_nan);model%kernel=make_kernel(0,1)
    select case(name)
    case('alpha_empty');allocate(model%alpha_values(0))
    case('alpha_count');model%alpha_values=[.1_dp,.2_dp]
    case('alpha_negative');model%alpha_values=[-.1_dp]
    case('alpha_nan');model%alpha_values=[bad]
    case('optimizer');model%optimizer='unsupported'
    case('restart');model%n_restarts_optimizer=-1
    case('seed');model%random_state=-1
    case('iterations');model%max_iter=0
    case('evaluations');model%max_fun=0
    case('tolerance');model%tol=-1
    case('nan_tolerance');model%ftol=bad
    case('target_count');model%n_targets=3
    case('negative_targets');model%n_targets=-1
    case('target_nan');target(1,2)=bad
    case('bounds_shape');kernel=make_kernel(0,1);copy=kernel%with_bounds(reshape([0._dp,1._dp],[1,2]));return
    case('bounds_order')
      kernel=rbf_covariance();copy=kernel%with_bounds(reshape([1._dp,0._dp],[1,2]));return
    case('bounds_nan');kernel=rbf_covariance();copy=kernel%with_bounds(reshape([bad,1._dp],[1,2]));return
    case('bounds_infinity')
      kernel=rbf_covariance();copy=kernel%with_bounds(reshape([ieee_value(0._dp,ieee_positive_inf), &
        ieee_value(0._dp,ieee_positive_inf)],[1,2]));return
    case('restart_infinite')
      kernel=rbf_covariance();model%kernel=kernel%with_bounds(reshape([ieee_value(0._dp,ieee_negative_inf), &
        1._dp],[1,2]));model%optimizer='fmin_l_bfgs_b';model%n_restarts_optimizer=1
    case('unfitted_likelihood');value=model%log_marginal_likelihood();return
    case('gradient_without_theta')
      call model%fit(data,target);value=model%log_marginal_likelihood(gradient=gradient);return
    case('predict_vector_multi');call model%fit(data,target);call vector(model%predict(data));return
    case('std_vector_multi');call model%fit(data,target);call vector(model%predict_std(data));return
    case('cov_vector_multi');call model%fit(data,target);call matrix(model%predict_cov(data));return
    case('score_weights')
      call model%fit(data,target);value=model%score(data,target,[1._dp]);return
    case('score_zero_weights')
      call model%fit(data,target);value=model%score(data,target,[0._dp,0._dp,0._dp,0._dp,0._dp]);return
    case('score_negative_weights')
      call model%fit(data,target);value=model%score(data,target,[1._dp,1._dp,-1._dp,1._dp,1._dp]);return
    case('score_nan_weights')
      call model%fit(data,target);value=model%score(data,target,[1._dp,1._dp,bad,1._dp,1._dp]);return
    case default;stop 2
    end select
    call model%fit(data,target)
  end subroutine
end program
