program reference_non_negative_factorization_driver
  use sklearn_decomposition, only: non_negative_factorization
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  real(dp), allocatable :: x(:,:),w(:,:),h(:,:),w0(:,:),h0(:,:)
  real(dp) :: beta,tol,aw,ah,l1
  integer :: n,p,k,rank,steps,seed,shuf,method,solver,fixed,it
  character(8), parameter :: methods(6)=[character(8)::'auto','random','custom','nndsvd','nndsvda','nndsvdar']
  character(2), parameter :: solvers(2)=['cd','mu']
  character(40) :: arg
  call get_command_argument(1,arg)
  if(len_trim(arg)>0) then
    allocate(x(4,3),h0(2,3),w0(4,2));x=1;h0=.4_dp;w0=.3_dp
    select case(trim(arg))
    case('missing_h')
      call non_negative_factorization(x,w,h,it,update_h=.false.)
    case('h_width')
      call non_negative_factorization(x,w,h,it,h0=h0(:,:2),update_h=.false.)
    case('h_rank')
      call non_negative_factorization(x,w,h,it,n_components=3,h0=h0,update_h=.false.)
    case('empty_h')
      call non_negative_factorization(x,w,h,it,h0=h0(:0,:),update_h=.false.)
    case('negative_h')
      h0(1,1)=-1;call non_negative_factorization(x,w,h,it,h0=h0,update_h=.false.)
    case('nan_h')
      h0(1,1)=ieee_value(0._dp,ieee_quiet_nan)
      call non_negative_factorization(x,w,h,it,h0=h0,update_h=.false.)
    case('zero_h')
      h0=0;call non_negative_factorization(x,w,h,it,h0=h0,update_h=.false.)
    case('long_init')
      call non_negative_factorization(x,w,h,it,init='nndsvdar_invalid')
    case('long_solver')
      call non_negative_factorization(x,w,h,it,solver='cd_invalid')
    case('custom_missing_w')
      call non_negative_factorization(x,w,h,it,init='custom',h0=h0)
    case('invalid_fixed_init')
      call non_negative_factorization(x,w,h,it,h0=h0,update_h=.false.,init='invalid')
    case('cd_beta')
      call non_negative_factorization(x,w,h,it,h0=h0,update_h=.false.,beta_loss=1._dp)
    case default
      error stop 'unknown rejection'
    end select
    error stop 'invalid input accepted'
  end if
  read(*,*) n,p,k,rank,steps,seed,shuf,method,solver,fixed,beta,tol,aw,ah,l1
  allocate(x(n,p),w0(n,rank),h0(rank,p));read(*,*) x;read(*,*) w0;read(*,*) h0
  call non_negative_factorization(x,w,h,it,n_components=k,w0=w0,h0=h0,init=methods(method+1), &
    update_h=fixed==0,solver=solvers(solver+1),beta_loss=beta,tol=tol,max_iter=steps, &
    alpha_w=aw,alpha_h=ah,l1_ratio=l1,random_state=seed,shuffle=shuf/=0)
  call matrix(w);call matrix(h);print *,1,1;print *,it
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *,shape(a)
    print '(*(es25.16e3,1x))',a
  end subroutine
end program
