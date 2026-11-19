program reference_spline_driver
  use sklearn, only: dp,spline_transformer
  implicit none
  type(spline_transformer) :: model
  real(dp), allocatable :: x(:,:),query(:,:),custom(:,:),fit_values(:,:),values(:,:)
  real(dp) :: low,high
  integer :: n,p,i,j,k,input,output
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),query(11,p),custom(5,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,12
    model=spline_transformer()
    do k=1,p
      low=minval(x(:,k)); high=maxval(x(:,k))
      custom(:,k)=low+(high-low)*[0._dp,.1_dp,.35_dp,.8_dp,1._dp]
      do i=1,11
        query(i,k)=low+(high-low)*(-.5_dp+real(i-1,dp)/5)
      end do
    end do
    select case(j)
    case(2); model%degree=1
    case(3); model%degree=2; model%include_bias=.false.
    case(4); model%knots='quantile'
    case(5); model%extrapolation='linear'
    case(6); model%extrapolation='continue'
    case(7); model%extrapolation='periodic'
    case(8); model%extrapolation='periodic'; model%include_bias=.false.
    case(10); model%extrapolation='periodic'; model%degree=2
    case(11)
      model%extrapolation='error'
      do k=1,p
        query(:,k)=max(minval(x(:,k)),min(maxval(x(:,k)),query(:,k)))
      end do
    case(12); model%degree=10; model%extrapolation='continue'
    end select
    if(j==9.or.j==10) then
      fit_values=model%fit_transform(x,knots=custom)
    else
      fit_values=model%fit_transform(x)
    end if
    values=model%transform(query); write(prefix,'(a,i0)') 'case',j
    call matrix('_train',fit_values); call matrix('_query',values)
    call matrix('_knots',model%knots_); call matrix('_base',model%base_knots_)
    call vector('_shape',real([model%n_features_in_,model%n_features_out_],dp))
  end do
  close(output)
contains
  subroutine vector(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') trim(prefix)//suffix,values
  end subroutine
  subroutine matrix(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:,:)
    call vector(suffix,reshape(values,[size(values)]))
  end subroutine
end program
