program reference_gp_kernels_driver
  use sklearn_gp_kernels
  use sklearn_gaussian_process, only: gaussian_process_regressor,gaussian_process_classifier
  use sklearn_common, only: dp
  use sklearn_bessel_k, only: log_besselk
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_positive_inf,ieee_quiet_nan
  implicit none
  type(gp_kernel) :: kernel,copy,empty
  type(gaussian_process_regressor) :: reg
  type(gaussian_process_classifier) :: classifier
  real(dp), allocatable :: x(:,:),q(:,:),yr(:),theta(:),gradient(:,:,:),vals(:)
  integer, allocatable :: yi(:)
  real(dp) :: nu,z,bad
  integer :: mode,code,n,p,m,config,i
  character(80) :: argument
  if(command_argument_count()>0) then
    call get_command_argument(1,argument);call reject(trim(argument));stop 2
  end if
  read(*,*) mode,code,n,p,m,config
  if(mode==3) then
    allocate(vals(n))
    do i=1,n
      read(*,*) nu,z;vals(i)=log_besselk(nu,z)
    end do
    call vector(vals);stop
  end if
  allocate(x(n,p),q(m,p),yr(n),yi(n));read(*,*) x;read(*,*) q;read(*,*) yr;read(*,*) yi
  kernel=make_kernel(code,p,config)
  select case(mode)
  case(0)
    call matrix(kernel%matrix(x));call matrix(kernel%matrix(x,q));call vector(kernel%diag(x))
    gradient=kernel%gradient(x);print *,shape(gradient)
    print '(*(es26.17e3,1x))',reshape(gradient,[size(gradient)])
    theta=kernel%theta();call vector(theta);copy=kernel%with_theta(theta);call matrix(copy%matrix(x))
    copy=kernel%with_theta(theta+.03_dp);call matrix(copy%matrix(x));call vector(copy%theta())
    call matrix(kernel%matrix(x))
  case(1)
    reg%kernel=kernel;reg%alpha=1;reg%normalize_y=mod(config,2)==1
    call vector(reg%predict(q));call vector(reg%predict_std(q));call matrix(reg%predict_cov(q))
    call reg%fit(x,yr);call vector(reg%predict(q));call vector(reg%predict_std(q));call matrix(reg%predict_cov(q))
    call vector([reg%log_marginal_likelihood_]);call vector(reg%alpha_);call matrix(reg%l_)
    reg%kernel=dot_product_covariance(4._dp);reg%amplitude=99;reg%length_scale=99;reg%noise_level=9
    call vector(reg%predict(q));call reg%fit(x,yr);call vector(reg%predict(q))
    call vector([reg%log_marginal_likelihood_])
  case(2)
    classifier%kernel=kernel;call classifier%fit(x,yi)
    call matrix(classifier%predict_proba(q));call vector(real(classifier%predict(q),dp))
    call vector([classifier%log_marginal_likelihood_]);call vector(classifier%log_marginal_likelihoods_)
    call matrix(classifier%pi_);call matrix(classifier%w_sqrt_);call matrix(classifier%alpha_)
    classifier%kernel=dot_product_covariance(4._dp);classifier%amplitude=99;classifier%length_scale=99
    call matrix(classifier%predict_proba(q));call classifier%fit(x,yi)
    call matrix(classifier%predict_proba(q));call vector([classifier%log_marginal_likelihood_])
  end select
contains
  function make_kernel(code,p,config) result(k)
    integer, intent(in) :: code,p,config
    type(gp_kernel) :: k
    real(dp) :: lengths(p)
    logical :: fixed,other
    integer :: j
    lengths=[(.4_dp+1.4_dp*real(j-1,dp)/max(1,p-1),j=1,p)]
    fixed=mod(config,3)==1;other=mod(config,3)==2
    select case(code)
    case(0);k=rbf_covariance(.7_dp,fixed)
    case(1);k=rbf_covariance(lengths,fixed)
    case(2);k=matern_covariance(1.2_dp,.5_dp,fixed)
    case(3);k=matern_covariance(lengths,1.5_dp,fixed)
    case(4);k=matern_covariance(1.2_dp,2.5_dp,fixed)
    case(5);k=matern_covariance(lengths,ieee_value(0._dp,ieee_positive_inf),fixed)
    case(6);k=matern_covariance(1.2_dp,.1_dp,fixed)
    case(7);k=matern_covariance(lengths,.7_dp,fixed)
    case(8);k=matern_covariance(1.2_dp,1.2_dp,fixed)
    case(9);k=matern_covariance(lengths,3.2_dp,fixed)
    case(10);k=matern_covariance(1.2_dp,5.3_dp,fixed)
    case(11);k=rational_quadratic_covariance(.8_dp,.3_dp,fixed_length=fixed,fixed_alpha=other)
    case(12);k=periodic_covariance(.9_dp,1.7_dp,fixed_length=fixed,fixed_periodicity=other)
    case(13);k=dot_product_covariance(.4_dp,fixed)
    case(14);k=constant_covariance(1.8_dp,fixed)
    case(15);k=white_covariance(.12_dp,fixed)
    case(16);k=1.3_dp*rbf_covariance(lengths,fixed)+white_covariance(.07_dp,other)
    case(17);k=(rbf_covariance(.7_dp,fixed)+dot_product_covariance(.4_dp,other))* &
      rational_quadratic_covariance(.8_dp,.3_dp,fixed_length=fixed,fixed_alpha=other)
    case(18);k=(periodic_covariance(.9_dp,1.7_dp)*matern_covariance(lengths,1.5_dp)+white_covariance(.02_dp))**2
    case(19);k=(.8_dp+rbf_covariance(1.3_dp,fixed))**1.7_dp
    case(20);k=constant_covariance(1.2_dp,.true.)*rbf_covariance(.8_dp,.true.)+white_covariance(.03_dp,.true.)
    case(21);k=rational_quadratic_covariance(.8_dp,.3_dp,fixed_alpha=.true.)+ &
      periodic_covariance(.9_dp,1.7_dp,fixed_length=.true.)+dot_product_covariance(.4_dp,.true.)
    case(22);k=dot_product_covariance(.4_dp,fixed)**2
    case(23);k=constant_covariance(0._dp,fixed)+white_covariance(0._dp,fixed)+dot_product_covariance(0._dp,fixed)
    case(24);k=rbf_covariance(.7_dp,fixed)**(-1._dp)
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
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: data(3,2),t(1)
    data=1;bad=ieee_value(0._dp,ieee_quiet_nan);t=0
    select case(name)
    case('empty');call matrix(empty%matrix(data))
    case('empty_expression');kernel=empty+rbf_covariance()
    case('empty_lengths');kernel=rbf_covariance([real(dp)::])
    case('zero_length');kernel=rbf_covariance(0._dp)
    case('negative_length');kernel=rbf_covariance([-1._dp,1._dp])
    case('nan_length');kernel=rbf_covariance(bad)
    case('nu_zero');kernel=matern_covariance(nu=0._dp)
    case('nu_nan');kernel=matern_covariance(nu=bad)
    case('alpha_zero');kernel=rational_quadratic_covariance(alpha=0._dp)
    case('period_zero');kernel=periodic_covariance(periodicity=0._dp)
    case('negative_constant');kernel=constant_covariance(-1._dp)
    case('nan_constant');kernel=constant_covariance(bad)
    case('negative_noise');kernel=white_covariance(-1._dp)
    case('negative_sigma');kernel=dot_product_covariance(-1._dp)
    case('nan_power');kernel=rbf_covariance()**bad
    case('length_count');kernel=rbf_covariance([1._dp,1._dp,1._dp]);call matrix(kernel%matrix(data))
    case('diag_length_count');kernel=rbf_covariance([1._dp,1._dp,1._dp]);call vector(kernel%diag(data))
    case('feature_count');kernel=rbf_covariance();call matrix(kernel%matrix(data,data(:,:1)))
    case('nan_X');data(1,1)=bad;kernel=rbf_covariance();call matrix(kernel%matrix(data))
    case('theta_count');kernel=rbf_covariance();copy=kernel%with_theta([0._dp,1._dp])
    case('theta_nan');kernel=rbf_covariance();copy=kernel%with_theta([bad])
    case('theta_overflow');kernel=rbf_covariance();copy=kernel%with_theta([1000._dp])
    case('theta_zero_length')
      kernel=rbf_covariance();copy=kernel%with_theta([-1000._dp]);call matrix(copy%matrix(data))
    case('fractional_negative')
      data=0;data(:,1)=[-1._dp,0._dp,1._dp];kernel=dot_product_covariance(0._dp)**.5_dp;call matrix(kernel%matrix(data))
    case('negative_zero');kernel=white_covariance(0._dp)**(-1._dp);call matrix(kernel%matrix(data))
    case('gp_nan_alpha');reg%alpha=bad;call reg%fit(data,[1._dp,2._dp,3._dp])
    case('gpc_kernel_length')
      classifier%kernel=rbf_covariance([1._dp,1._dp,1._dp]);call classifier%fit(data,[0,1,0])
    case default;stop 2
    end select
  end subroutine
end program
