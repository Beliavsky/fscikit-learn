program reference_cluster_information_driver
  use sklearn_metrics, only: normalized_mutual_info_score,adjusted_mutual_info_score, &
    fowlkes_mallows_score,contingency_matrix,pair_confusion_matrix,entropy,expected_mutual_information, &
    homogeneity_completeness_v_measure,homogeneity_score,completeness_score,v_measure_score
  use sklearn_common, only: dp
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  integer, allocatable :: a(:),b(:),manual(:,:)
  integer(int64), allocatable :: table(:,:)
  integer(int64) :: pairs(2,2)
  real(dp), allocatable :: smooth(:,:)
  real(dp) :: h,c,v,beta,eps,normal(4),adjusted(4)
  character(10), parameter :: methods(*)=[character(10) :: 'min','geometric','arithmetic','max']
  character(512) :: file
  integer :: input,output,n,i,mode,r,s
  call get_command_argument(1,file); open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file); open(newunit=output,file=trim(file),status='replace')
  read(input,*) mode,n,beta,eps
  if(mode==2) then
    read(input,*) r,s; allocate(manual(r,s))
    do i=1,r
      read(input,*) manual(i,:)
    end do
    call emit('expected',[expected_mutual_information(manual,n),expected_mutual_information(int(manual,int64),n)])
  else
    allocate(a(n),b(n))
    if(n>0) then
      read(input,*) a; read(input,*) b
    end if
    table=contingency_matrix(a,b); smooth=contingency_matrix(a,b,eps=eps)
    call emit_integer('table',reshape(table,[size(table)]))
    call emit_integer('shape',int(shape(table),int64)); call emit('smoothed',reshape(smooth,[size(smooth)]))
    pairs=pair_confusion_matrix(a,b); call emit_integer('pairs',reshape(pairs,[4]))
    call emit('entropy',[entropy(a),entropy(b)])
    call emit('fowlkes',[fowlkes_mallows_score(a,b),fowlkes_mallows_score(b,a)])
    do i=1,4
      normal(i)=normalized_mutual_info_score(a,b,methods(i))
      adjusted(i)=adjusted_mutual_info_score(a,b,methods(i))
    end do
    call emit('normal',normal); call emit('adjusted',adjusted)
    call emit('defaults',[normalized_mutual_info_score(a,b),adjusted_mutual_info_score(a,b)])
    call homogeneity_completeness_v_measure(a,b,h,c,v,beta)
    call emit('hcv',[h,c,v])
    call emit('wrappers',[homogeneity_score(a,b),completeness_score(a,b),v_measure_score(a,b,beta=beta)])
    if(n>0) call emit('expected',[expected_mutual_information(table,n)])
    call emit_integer('preserved',int([a,b],int64))
  end if
  close(input); close(output)
contains
  subroutine emit_integer(name,values)
    character(*), intent(in) :: name
    integer(int64), intent(in) :: values(:)
    write(output,'(a,*(1x,i0))') name,values
  end subroutine
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es26.17e3))') name,values
  end subroutine
end program
