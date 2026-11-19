program reference_power_transform_driver
  use sklearn
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(power_transformer) :: model
  real(dp), allocatable :: x(:,:),train(:,:),query(:,:),z(:,:),back(:,:)
  integer :: input,output,n,p,i,j
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,8
    train=x; query=x+0.17_dp; model%method='yeo-johnson'; model%standardize=mod(j,2)==1
    select case(j)
    case(3,4)
      train=exp(x); query=exp(x+0.17_dp); model%method='box-cox'
    case(5,6)
      train(2,1)=ieee_value(0._dp,ieee_quiet_nan); query(3,1)=train(2,1)
    case(7,8)
      train(:,p)=3; query(:,p)=3.17_dp
    end select
    z=model%fit_transform(train); back=model%inverse_transform(z)
    write(prefix,'(a,i0)') 'power',j
    call vector(trim(prefix)//'_lambda',model%lambdas_)
    call vector(trim(prefix)//'_train',reshape(z,[size(z)]))
    call vector(trim(prefix)//'_query',reshape(model%transform(query),[size(query)]))
    call vector(trim(prefix)//'_inverse',reshape(back,[size(back)]))
  end do
  close(output)
contains
  subroutine vector(name,v)
    character(*), intent(in) :: name
    real(dp), intent(in) :: v(:)
    write(output,'(a,*(1x,es25.16e3))') name,v
  end subroutine
end program
