program reference_bicluster_datasets_driver
  use sklearn_common, only: dp
  use sklearn_datasets, only: make_biclusters,make_checkerboard
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  integer :: cases,index,kind,n,p,kr,kc,seed,shuf
  real(dp) :: noise,low,high,nan
  real(dp), allocatable :: x(:,:),other(:,:)
  logical, allocatable :: rows(:,:),cols(:,:),r2(:,:),c2(:,:)
  character(64) :: mode
  nan=ieee_value(0._dp,ieee_quiet_nan)
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    call reject(trim(mode));error stop 'invalid bicluster generator input accepted'
  end if
  read(*,*) cases
  do index=1,cases
    read(*,*) kind,n,p,kr,kc,seed,shuf,noise,low,high
    select case(kind)
    case(0)
      call make_biclusters([n,p],kr,x,rows,cols,noise,low,high,shuf==1,seed)
      call make_biclusters([n,p],kr,other,r2,c2,noise,low,high,shuf==1,seed)
    case(1)
      call make_checkerboard([n,p],kr,x,rows,cols,noise,low,high,shuf==1,seed)
      call make_checkerboard([n,p],kr,other,r2,c2,noise,low,high,shuf==1,seed)
    case(2)
      call make_checkerboard([n,p],[kr,kc],x,rows,cols,noise,low,high,shuf==1,seed)
      call make_checkerboard([n,p],[kr,kc],other,r2,c2,noise,low,high,shuf==1,seed)
    end select
    write(*,'(a,1x,i0)') 'BEGIN',index
    call matrix('data',x);call matrix('rows',real(merge(1,0,rows),dp));call matrix('cols',real(merge(1,0,cols),dp))
    call vector('metadata',real([size(x,1),size(x,2),size(rows,1)],dp))
    call vector('repeat_identity',[merge(1._dp,0._dp,all_equal(x,other).and.all(rows.eqv.r2).and.all(cols.eqv.c2))])
    write(*,'(a)') 'END';flush(6)
  end do
contains
  logical function all_equal(a,b) result(ok)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    real(dp), intent(in) :: a(:,:),b(:,:)
    integer :: i,j
    ok=.true.
    do i=1,size(a,1)
      do j=1,size(a,2)
        if(ieee_is_nan(a(i,j)).and.ieee_is_nan(b(i,j))) cycle
        if(a(i,j)/=b(i,j)) ok=.false.
      end do
    end do
  end function
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    integer :: j
    write(*,'(a,1x,i0)',advance='no') name,size(a)
    do j=1,size(a)
      write(*,'(1x,es25.17e3)',advance='no') a(j)
    end do
    write(*,*)
  end subroutine
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(transpose(a),[size(a)]))
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    select case(name)
    case('shape');call make_biclusters([1,2,3],2,x,rows,cols)
    case('negative_shape');call make_biclusters([-1,2],2,x,rows,cols)
    case('clusters');call make_biclusters([2,3],0,x,rows,cols)
    case('pair_shape');call make_checkerboard([2,3],[2],x,rows,cols)
    case('pair_clusters');call make_checkerboard([2,3],[2,0],x,rows,cols)
    case('block_overflow');call make_checkerboard([1,1],[huge(1),2],x,rows,cols)
    case('noise');call make_biclusters([2,3],2,x,rows,cols,noise=-1._dp)
    case('nan_noise');call make_checkerboard([2,3],2,x,rows,cols,noise=nan)
    case('inf_noise');call make_biclusters([2,3],2,x,rows,cols,noise=ieee_value(0._dp,ieee_positive_inf))
    case('nan_min');call make_biclusters([2,3],2,x,rows,cols,minval=nan)
    case('inf_max');call make_checkerboard([2,3],2,x,rows,cols,maxval=ieee_value(0._dp,ieee_positive_inf))
    case('bounds');call make_biclusters([2,3],2,x,rows,cols,minval=2._dp,maxval=1._dp)
    case('range');call make_checkerboard([2,3],2,x,rows,cols,minval=-1.e308_dp,maxval=1.e308_dp)
    case('seed');call make_biclusters([2,3],2,x,rows,cols,random_state=-1)
    case default;error stop 'unknown rejection'
    end select
  end subroutine
end program
