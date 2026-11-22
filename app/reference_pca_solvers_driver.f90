program reference_pca_solvers_driver
  use sklearn_common, only: dp
  use sklearn_decomposition, only: pca
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_randomized_svd_core, only: randomized_svd
  implicit none
  type(pca) :: model
  type(legacy_random_state) :: rng
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:),s(:),u(:,:),vt(:,:)
  integer :: n,p,m,k,mode,white,solver,power,extra,normalizer,seed,flip
  real(dp) :: fraction,next_normal,next_uniform
  character(8), parameter :: norms(4)=[character(8)::'auto','none','LU','QR']
  character(16), parameter :: solvers(4)=[character(16)::'full','randomized','covariance_eigh','auto']
  character(80) :: argument
  if(command_argument_count()>0) then
    call get_command_argument(1,argument)
    if(trim(argument)=='core') then
      read(*,*) n,p,k,power,extra,normalizer,seed,flip
      allocate(x(n,p));read(*,*) x;call rng%seed(seed)
      if(power==-1) then
        call randomized_svd(x,k,rng,s,u,vt,n_oversamples=extra, &
          power_iteration_normalizer=norms(normalizer+1),flip_sign=flip/=0)
      else
        call randomized_svd(x,k,rng,s,u,vt,n_iter=power,n_oversamples=extra, &
          power_iteration_normalizer=norms(normalizer+1),flip_sign=flip/=0)
      end if
      next_normal=rng%normal();next_uniform=rng%uniform()
      call vector(s);call matrix(u);call matrix(vt);call vector([next_normal,next_uniform]);stop
    end if
    call reject(trim(argument));stop 2
  end if
  read(*,*) n,p,m,k,mode,fraction,white,solver,power,extra,normalizer,seed
  allocate(x(n,p),q(m,p));read(*,*) x;read(*,*) q
  model%n_components=k;model%variance_fraction=fraction;model%whiten=white/=0
  model%svd_solver=solvers(solver+1);model%iterated_power=power;model%n_oversamples=extra
  model%power_iteration_normalizer=norms(normalizer+1);model%random_state=seed
  select case(mode)
  case(1);model%dimension_mode='variance'
  case(2);model%dimension_mode='mle'
  case(3);model%dimension_mode='zero'
  end select
  z=model%fit_transform(x)
  call matrix(model%components_);call vector(model%explained_variance_);call vector(model%explained_variance_ratio_)
  call vector(model%singular_values_);call vector(model%mean_);call vector([model%noise_variance_])
  solver=findloc(solvers,model%fit_svd_solver_,dim=1)-1
  call vector(real([model%n_samples_,model%n_components_,solver],dp))
  call matrix(z);call matrix(model%transform(x));call matrix(model%transform(q))
  call matrix(model%inverse_transform(model%transform(q)));call matrix(model%get_covariance())
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
    real(dp) :: data(6,2)
    data(:,1)=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp];data(:,2)=[0._dp,1._dp,0._dp,2._dp,1._dp,3._dp]
    select case(name)
    case('solver');model%svd_solver='unsupported'
    case('power');model%iterated_power=-2
    case('oversampling');model%n_oversamples=0
    case('seed');model%random_state=-1
    case('normalizer');model%power_iteration_normalizer='unknown'
    case('random_fraction');model%svd_solver='randomized';model%dimension_mode='variance'
    case('random_mle');model%svd_solver='randomized';model%dimension_mode='mle'
    case('random_zero');model%svd_solver='randomized';model%dimension_mode='zero'
    case('capacity');model%svd_solver='randomized';model%n_oversamples=huge(1)
    case('core_oversampling');call rng%seed(0);call randomized_svd(data,1,rng,s,u,vt,n_oversamples=-1);return
    case('core_power');call rng%seed(0);call randomized_svd(data,1,rng,s,u,vt,n_iter=-1);return
    case('core_normalizer')
      call rng%seed(0);call randomized_svd(data,1,rng,s,u,vt,power_iteration_normalizer='unknown');return
    case('core_capacity');call rng%seed(0);call randomized_svd(data,1,rng,s,u,vt,n_oversamples=huge(1));return
    case default;stop 2
    end select
    call model%fit(data)
  end subroutine
end program
