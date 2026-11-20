program reference_univariate_selection_driver
  use sklearn_univariate_selection
  use sklearn_common, only: dp
  implicit none
  type(select_percentile), target :: percentile
  type(select_fpr), target :: fpr
  type(select_fdr), target :: fdr
  type(select_fwe), target :: fwe
  type(generic_univariate_select), target :: generic
  class(generic_univariate_select), pointer :: model
  real(dp), allocatable :: x(:,:),yr(:),scores(:),pvalues(:),z(:,:),restored(:,:)
  integer, allocatable :: yc(:),indices(:)
  logical, allocatable :: mask(:)
  integer :: input,output,n,p,nops,kind,task,k,i,has_p
  real(dp) :: value
  character(512) :: file
  character(24) :: prefix
  call get_command_argument(1,file); open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file); open(newunit=output,file=trim(file),status='replace')
  read(input,*) nops
  do k=1,nops
    read(input,*) kind,task,n,p,value,has_p
    select case(kind)
    case(1); model=>percentile
    case(2); model=>fpr
    case(3); model=>fdr
    case(4); model=>fwe
    case default; model=>generic
    end select
    model%percentile=value; model%alpha=value; model%param=value
    read(input,*) model%mode,model%score_func,model%center,model%k_all
    if(allocated(x)) deallocate(x,yr,yc)
    allocate(x(n,p),yr(n),yc(n))
    if(task==3) then
      if(allocated(scores)) deallocate(scores)
      allocate(scores(p)); read(input,*) scores
      if(allocated(pvalues)) deallocate(pvalues)
      if(has_p/=0) then
        allocate(pvalues(p)); read(input,*) pvalues
        call model%fit_scores(scores,pvalues)
      else
        call model%fit_scores(scores)
      end if
    end if
    do i=1,n
      read(input,*) x(i,:),yr(i)
    end do
    yc=nint(yr)
    if(task==1) call model%fit(x,yc)
    if(task==2) call model%fit(x,yr)
    write(prefix,'(i0,a)') k,'_'
    call emit('scores',model%scores_)
    if(allocated(model%pvalues_)) call emit('pvalues',model%pvalues_)
    mask=model%get_support(); call emit('mask',real(merge(1,0,mask),dp))
    indices=model%get_support_indices(); call emit('indices',real(indices-1,dp))
    z=model%transform(x); call emit('transform',reshape(z,[size(z)]))
    restored=model%inverse_transform(z); call emit('inverse',reshape(restored,[size(restored)]))
    call emit('meta',real([model%n_features_in_,count(mask)],dp))
    ! Controls remain live after fitting, as in source get_support/transform.
    model%percentile=0; model%alpha=0; model%param=0; model%k_all=.false.
    mask=model%get_support(); call emit('changed_mask',real(merge(1,0,mask),dp))
    z=model%transform(x); restored=model%inverse_transform(z)
    call emit('changed_inverse',reshape(restored,[size(restored)]))
  end do
  close(input); close(output)
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es26.17e3))') trim(prefix)//name,values
  end subroutine
end program
