program reference_gp_sampling_driver
  use sklearn_gaussian_process, only: gaussian_process_regressor
  use sklearn_gp_kernels
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(gaussian_process_regressor) :: model
  type(gp_kernel) :: kernel
  real(dp), allocatable :: x(:,:),q(:,:),y(:,:),samples(:,:,:),single(:,:),flat(:,:),mean(:),centered(:,:)
  real(dp), allocatable :: predicted(:,:),cov(:,:,:),again(:,:,:),empty(:,:,:)
  integer :: code,n,p,m,targets,draws,seed,config,i,total
  real(dp) :: agreement,bad
  character(80) :: argument
  if(command_argument_count()>0) then
    call get_command_argument(1,argument);call reject(trim(argument));stop 2
  end if
  read(*,*) code,n,p,m,targets,draws,seed,config
  allocate(x(n,p),q(m,p),y(n,targets));read(*,*) x;read(*,*) q;read(*,*) y
  model%kernel=make_kernel(code,p);model%n_targets=targets;model%random_state=99
  model%normalize_y=config>=3;model%alpha=.04_dp
  if(config==5) model%alpha_values=[(.02_dp+.01_dp*i,i=1,n)]
  if(config>0) then
    if(config==1.and.targets==1) then
      call model%fit(x,y(:,1))
    else
      call model%fit(x,y)
    end if
  end if
  if(config==4) model%kernel=constant_covariance(0._dp,.true.)
  predicted=model%predict_multi(q);cov=model%predict_cov_multi(q)
  call matrix(predicted);call tensor(cov)
  samples=model%sample_y_multi(q,draws,seed)
  if(draws<=1000) then
    call tensor(samples)
    again=model%sample_y_multi(q,draws,seed)
    call require(all(samples==again),'GP sampling repeated calls differ')
  else
    total=m*targets;flat=reshape(samples,[total,draws]);mean=sum(flat,dim=2)/draws
    centered=flat-spread(mean,2,draws);call vector(mean)
    call matrix(matmul(centered,transpose(centered))/(draws-1))
  end if
  agreement=0
  if(targets==1.and.draws<=1000) then
    single=model%sample_y(q,draws,seed)
    if(draws>0) agreement=maxval(abs(single-samples(:,1,:)))
  end if
  call vector([agreement]);call matrix(model%predict_multi(q));call tensor(model%predict_cov_multi(q))
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
    case(9);k=constant_covariance(1.7_dp,.true.)
    case(10);k=white_covariance(.2_dp,.true.)
    case(11);k=constant_covariance(0._dp,.true.)
    case(12);k=rbf_covariance(.8_dp,.true.)
    case(13);k=constant_covariance(1.e-240_dp,.true.)
    case(14);k=constant_covariance(1.e240_dp,.true.)
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
    real(dp) :: data(2,1)
    data(:,1)=[-1._dp,1._dp];model%kernel=rbf_covariance(.8_dp)
    bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('negative_count');samples=model%sample_y_multi(data,-1)
    case('negative_seed');samples=model%sample_y_multi(data,1,-1)
    case('multiple_targets');model%n_targets=2;single=model%sample_y(data)
    case('capacity');model%n_targets=2;samples=model%sample_y_multi(data,huge(1))
    case('invalid_targets');model%n_targets=-1;samples=model%sample_y_multi(data)
    case('nan_query');data(1,1)=bad;samples=model%sample_y_multi(data)
    case('wrong_features')
      call model%fit(data,[1._dp,2._dp]);samples=model%sample_y_multi(reshape([1._dp,2._dp],[1,2]))
    case('nonfinite_moments')
      call model%fit(data,[1._dp,2._dp]);model%y_train_mean_=ieee_value(0._dp,ieee_positive_inf)
      samples=model%sample_y_multi(data)
    case default;stop 2
    end select
  end subroutine
end program
