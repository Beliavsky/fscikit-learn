program reference_factor_options_driver
  use sklearn_common, only: dp
  use sklearn_factor_analysis, only: factor_analysis
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(factor_analysis) :: model
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:),raw(:,:),noise(:)
  integer :: n,p,m,k,solver,rotation,power,seed,limit,init
  real(dp) :: tolerance,bad
  character(80) :: argument
  if(command_argument_count()>0) then
    call get_command_argument(1,argument);call reject(trim(argument));stop 2
  end if
  read(*,*) n,p,m,k,solver,rotation,power,seed,limit,tolerance,init
  allocate(x(n,p),q(m,p));read(*,*) x;read(*,*) q
  model%n_components=k;model%max_iter=limit;model%tol=tolerance
  model%random_state=seed;model%iterated_power=power
  if(solver==1) model%svd_method='randomized'
  if(rotation==1) model%rotation='varimax'
  if(rotation==2) model%rotation='quartimax'
  if(init/=0) then
    allocate(noise(p));read(*,*) noise;model%noise_variance_init=noise
  end if
  call model%fit(x);z=model%transform(q)
  raw=matmul(model%rotation_matrix_,model%components_)
  call matrix(model%components_);call matrix(raw);call vector(model%mean_);call vector(model%noise_variance_)
  call vector(model%loglike_);call vector(real([model%n_iter_,merge(1,0,model%converged_)],dp))
  call matrix(model%rotation_matrix_)
  call vector(real([model%n_rotation_iter_,merge(1,0,model%rotation_converged_)],dp))
  call matrix(z);call matrix(model%get_covariance());call matrix(model%get_precision())
  call vector(model%score_samples(q));call vector([model%score(q)]);call matrix(model%transform(x))
contains
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    print *,size(a);print '(*(es26.17e3,1x))',a
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *,shape(a);print '(*(es26.17e3,1x))',reshape(a,[size(a)])
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: data(5,2)
    data(:,1)=[1._dp,2._dp,3._dp,4._dp,5._dp];data(:,2)=[1._dp,0._dp,2._dp,1._dp,3._dp]
    bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('rotation');model%rotation='unsupported'
    case('solver');model%svd_method='unsupported'
    case('power');model%iterated_power=-1
    case('seed');model%random_state=-1
    case('iterations');model%max_iter=0
    case('negative_tol');model%tol=-1
    case('nan_tol');model%tol=bad
    case('infinite_tol');model%tol=ieee_value(0._dp,ieee_positive_inf)
    case('noise_nan');model%noise_variance_init=[1._dp,bad]
    case('noise_infinite');model%noise_variance_init=[1._dp,ieee_value(0._dp,ieee_positive_inf)]
    case('noise_shape');model%noise_variance_init=[1._dp]
    case('noise_zero');model%noise_variance_init=[1._dp,0._dp]
    case('components');model%n_components=3
    case('unfitted');z=model%transform(data);return
    case('features')
      call model%fit(data);z=model%transform(reshape([1._dp,2._dp,3._dp],[1,3]));return
    case default;stop 2
    end select
    call model%fit(data)
  end subroutine
end program
