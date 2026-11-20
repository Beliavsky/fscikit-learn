program reference_sgd_driver
  use sklearn, only: dp,sgd_regressor
  implicit none
  type(sgd_regressor) :: model
  real(dp), allocatable :: x(:,:),y(:),w(:),weights(:)
  integer :: input,output,n,p,i,j
  character(512) :: path
  character(24) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n),w(n))
  do i=1,n
    read(input,*) x(i,:),y(i),w(i)
  end do
  close(input); call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,12
    model=sgd_regressor(); model%max_iter=7; model%tol=-1; model%eta0=.03_dp
    model%alpha=.01_dp; model%epsilon=.2_dp; weights=w
    select case(j)
    case(2); model%penalty='none'; model%learning_rate='constant'
    case(3); model%loss='huber'
    case(4); model%loss='epsilon_insensitive'
    case(5); model%loss='squared_epsilon_insensitive'
    case(6); model%average=1
    case(7); model%average=10
    case(8); model%average=1
    case(9); model%max_iter=300; model%tol=1.e-4_dp
    case(10); model%fit_intercept=.false.
    case(11); model%learning_rate='constant'; model%eta0=.1_dp; model%alpha=20
    case(12); weights=0
    end select
    if(j==8) then
      call model%partial_fit(x(:9,:),y(:9),weights(:9))
      call model%partial_fit(x(10:10,:),y(10:10),weights(10:10))
      call model%partial_fit(x(11:,:),y(11:),weights(11:))
    else
      call model%fit(x,y,sample_weight=weights)
    end if
    write(prefix,'(a,i0)') 'case',j
    call vector('_coef',model%coef_); call vector('_intercept',[model%intercept_])
    call vector('_train',model%predict(x)); call vector('_query',model%predict(x+.137_dp))
    call vector('_score',[model%score(x,y)]); call vector('_clock',[real(model%n_iter_,dp),model%t_])
  end do
  close(output)
contains
  subroutine vector(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') trim(prefix)//suffix,values
  end subroutine
end program
