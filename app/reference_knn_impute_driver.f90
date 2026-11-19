program reference_knn_impute_driver
  use sklearn, only: dp,knn_imputer,missing_indicator
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(knn_imputer) :: model
  type(missing_indicator) :: indicator
  real(dp), allocatable :: x(:,:),q(:,:),result(:,:),fitted(:,:)
  logical, allocatable :: mask(:,:)
  integer :: n,p,m,i,j,k,input,output
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,m; allocate(x(n,p),q(m,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  do i=1,m
    read(input,*) q(i,:)
  end do
  close(input)
  where(x==-9999._dp) x=ieee_value(0._dp,ieee_quiet_nan)
  where(q==-9999._dp) q=ieee_value(0._dp,ieee_quiet_nan)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,8
    model=knn_imputer()
    select case(j)
    case(1); model%n_neighbors=1
    case(2); model%n_neighbors=3
    case(3); model%n_neighbors=3; model%weights='distance'
    case(4); model%n_neighbors=n+10; model%weights='distance'
    case(5); model%add_indicator=.true.
    case(6); model%keep_empty_features=.true.
    case(7); model%keep_empty_features=.true.; model%add_indicator=.true.; model%weights='distance'
    case(8); model%n_neighbors=n+10; model%add_indicator=.true.
    end select
    fitted=model%fit_transform(x); result=model%transform(q)
    write(prefix,'(a,i0)') 'knn',j
    call matrix('_train',fitted); call matrix('_query',result)
    call vector('_shape',real([size(fitted,2),size(result,2)],dp))
    call vector('_valid',merge(1._dp,0._dp,model%valid_mask_))
    call vector('_features',real(model%indicator_%features_,dp))
  end do
  do j=1,3
    indicator=missing_indicator(); indicator%error_on_new=.false.
    if(j==2) indicator%features='all'
    if(j==3) indicator%error_on_new=.true.
    write(prefix,'(a,i0)') 'indicator',j
    mask=indicator%fit_transform(x); call matrix('_train',merge(1._dp,0._dp,mask))
    ! Strict mode uses the training data to avoid introducing new missing columns.
    if(j==3) then
      mask=indicator%transform(x)
    else
      mask=indicator%transform(q)
    end if
    call matrix('_query',merge(1._dp,0._dp,mask))
    call vector('_features',real(indicator%features_,dp))
    call vector('_shape',real(shape(mask),dp))
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
