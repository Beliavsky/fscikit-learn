program reference_gpc_ovo_driver
  use sklearn_gp_classification, only: gaussian_process_classifier
  use sklearn_gp_kernels
  use sklearn_common, only: dp
  implicit none
  type(gaussian_process_classifier) :: model
  type(gp_kernel) :: kernel
  real(dp), allocatable :: x(:,:),q(:,:),bounds(:,:),theta(:,:),packed(:),gradient(:),probability(:,:)
  integer, allocatable :: labels(:)
  integer :: code,n,p,m,seed,opt,restarts,warm,limit,nb,i,d,heads,r
  real(dp) :: value
  character(80) :: argument
  if(command_argument_count()>0) then
    call get_command_argument(1,argument);call reject(trim(argument));stop 2
  end if
  read(*,*) code,n,p,m,seed,opt,restarts,warm,limit,nb
  allocate(x(n,p),q(m,p),labels(n));read(*,*) x;read(*,*) q;read(*,*) labels
  kernel=make_kernel(code,p)
  if(nb>=0) then
    allocate(bounds(nb,2));if(nb>0) read(*,*) bounds
    kernel=kernel%with_bounds(bounds)
  end if
  model%kernel=kernel;model%multi_class='one_vs_one'
  model%random_state=seed;model%n_restarts_optimizer=restarts
  model%warm_start=warm/=0;model%max_iter_predict=limit
  if(opt/=0) model%optimizer='fmin_l_bfgs_b'
  call model%fit(x,labels);heads=size(model%kernels_);d=kernel%n_dims();allocate(theta(d,heads))
  do i=1,heads
    theta(:,i)=model%kernels_(i)%theta()
  end do
  call vector(real(model%predict(q),dp));call vector([model%log_marginal_likelihood()])
  call matrix(real(model%class_pairs_,dp));call vector(real(model%n_training_samples_,dp))
  call matrix(theta);call vector(model%log_marginal_likelihoods_)
  do i=1,heads
    r=model%n_training_samples_(i)
    call vector(real(model%training_indices_(:r,i),dp));call vector(model%pi_(:r,i))
    call vector(model%w_sqrt_(:r,i));call matrix(model%l_(:r,:r,i));call vector(model%alpha_(:r,i))
    call vector(model%posterior_mode_(:r,i));call vector(model%mode_cache_(:r,i))
    call vector(model%fit_initial_cache_(:r,i));call matrix(model%optimization_points_(:,:,i))
    call vector(model%optimization_values_(:,i));call matrix(model%optimization_eval_cache_(:r,:,i))
    call matrix(model%optimization_initial_points_(:,:,i))
  end do
  value=model%log_marginal_likelihood(theta(:,1)+.04_dp);call vector([value])
  do i=1,heads
    call vector(model%mode_cache_(:model%n_training_samples_(i),i))
  end do
  call vector(real(model%predict(q),dp))
  packed=reshape(theta,[size(theta)])+.02_dp
  value=model%log_marginal_likelihood(packed,clone_kernel=.false.);call vector([value])
  do i=1,heads
    call vector(model%kernels_(i)%theta());call vector(model%mode_cache_(:model%n_training_samples_(i),i))
  end do
  call vector(real(model%predict(q),dp));call vector([model%log_marginal_likelihood()])
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
    data(:,1)=[-1._dp,-.6_dp,-.2_dp,.2_dp,.6_dp,1._dp];y=[-4,-4,9,9,17,17]
    model%kernel=make_kernel(0,1);model%multi_class='one_vs_one'
    select case(name)
    case('mode');model%multi_class='unsupported'
    case('binary_mode');model%multi_class='unsupported';y=[-4,-4,-4,9,9,9]
    case('unfitted');labels=model%predict(data);return
    case('probability');call model%fit(data,y);probability=model%predict_proba(data);return
    case('theta');call model%fit(data,y);value=model%log_marginal_likelihood([0._dp]);return
    case('gradient')
      call model%fit(data,y);value=model%log_marginal_likelihood(model%kernels_(1)%theta(),gradient);return
    case default;stop 2
    end select
    call model%fit(data,y)
  end subroutine
end program
