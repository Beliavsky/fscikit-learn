program reference_sample_generators_driver
  use sklearn_datasets
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  real(dp), allocatable :: x(:,:),y(:),center(:)
  integer, allocatable :: labels(:)
  integer :: family,n,p,seed,config,i
  real(dp) :: noise,variance,bad
  character(80) :: argument
  if(command_argument_count()>0) then
    call get_command_argument(1,argument);call reject(trim(argument));stop 2
  end if
  read(*,*) family,n,p,seed,noise,config
  select case(family)
  case(0);call make_friedman1(n,p,x,y,noise,seed)
  case(1);call make_friedman2(n,x,y,noise,seed)
  case(2);call make_friedman3(n,x,y,noise,seed)
  case(3);call make_sparse_uncorrelated(n,p,x,y,seed)
  case(4);call make_s_curve(n,x,y,noise,seed)
  case(5);call make_swiss_roll(n,x,y,noise,seed,hole=mod(config,2)==1)
  case(6)
    allocate(center(p));read(*,*) center
    variance=noise
    call make_gaussian_quantiles(n,p,x,labels,n_classes=config,mean=center,cov=variance, &
      shuffle=mod(seed,2)==0,random_state=seed)
    y=real(labels,dp)
  case default;stop 2
  end select
  print *,size(x,1),size(x,2)
  do i=1,size(x,2)
    print '(*(es26.17e3,1x))',x(:,i)
  end do
  print '(*(es26.17e3,1x))',y
contains
  subroutine reject(name)
    character(*), intent(in) :: name
    bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('friedman1_samples');call make_friedman1(0,5,x,y)
    case('friedman2_samples');call make_friedman2(0,x,y)
    case('friedman3_samples');call make_friedman3(0,x,y)
    case('sparse_samples');call make_sparse_uncorrelated(0,4,x,y)
    case('s_curve_samples');call make_s_curve(0,x,y)
    case('swiss_samples');call make_swiss_roll(0,x,y)
    case('quantile_samples');call make_gaussian_quantiles(0,2,x,labels)
    case('friedman1_features');call make_friedman1(2,4,x,y)
    case('sparse_features');call make_sparse_uncorrelated(2,3,x,y)
    case('negative_noise');call make_friedman2(2,x,y,noise=-1._dp)
    case('nan_noise');call make_s_curve(2,x,y,noise=bad)
    case('infinite_noise');call make_swiss_roll(2,x,y,noise=ieee_value(0._dp,ieee_positive_inf))
    case('seed');call make_friedman3(2,x,y,random_state=-1)
    case('quantile_features');call make_gaussian_quantiles(3,0,x,labels)
    case('quantile_classes');call make_gaussian_quantiles(3,2,x,labels,n_classes=0)
    case('quantile_too_many_classes');call make_gaussian_quantiles(3,2,x,labels,n_classes=4)
    case('quantile_cov');call make_gaussian_quantiles(3,2,x,labels,cov=-1._dp)
    case('quantile_nan_cov');call make_gaussian_quantiles(3,2,x,labels,cov=bad)
    case('quantile_infinite_cov')
      call make_gaussian_quantiles(3,2,x,labels,cov=ieee_value(0._dp,ieee_positive_inf))
    case('quantile_mean_length');call make_gaussian_quantiles(3,2,x,labels,mean=[1._dp])
    case('quantile_mean_nan');call make_gaussian_quantiles(3,2,x,labels,mean=[0._dp,bad])
    case default;stop 2
    end select
  end subroutine
end program
