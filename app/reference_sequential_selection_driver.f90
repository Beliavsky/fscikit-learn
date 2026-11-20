program reference_sequential_selection_driver
  use sklearn, only: dp,sequential_feature_selector,linear_regression,ridge,elastic_net,lasso
  implicit none
  type(sequential_feature_selector) :: model
  type(linear_regression) :: ols
  type(ridge) :: reg
  type(elastic_net) :: elastic
  type(lasso) :: sparse
  real(dp), allocatable :: x(:,:),y(:),z(:,:)
  integer :: n,p,i,j,input,output
  character(512) :: path
  character(24) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n))
  do i=1,n
    read(input,*) x(i,:),y(i)
  end do
  close(input); call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  reg%alpha=2._dp; elastic%alpha=.07_dp; elastic%l1_ratio=.35_dp; elastic%tol=1.e-11_dp
  sparse%alpha=.1_dp; sparse%tol=1.e-11_dp
  do j=1,15
    model=sequential_feature_selector(); model%cv=4; model%n_features_to_select=2
    select case(j)
    case(2,10,12,13); model%direction='backward'
    case(3); model%n_features_to_select=0
    case(4); model%n_features_to_select=0; model%direction='backward'
    case(5); model%n_features_to_select=0; model%use_tol=.true.; model%tol=.01_dp
    case(6)
      model%n_features_to_select=0; model%direction='backward'; model%use_tol=.true.; model%tol=-.01_dp
    case(7); model%n_features_to_select=0; model%use_tol=.true.; model%tol=1.e6_dp
    case(8)
      model%n_features_to_select=0; model%direction='backward'; model%use_tol=.true.; model%tol=1.e6_dp
    case(14); model%n_features_to_select=1
    case(15); model%n_features_to_select=p-1; model%direction='backward'
    end select
    if(j==13) then
      model%use_tol=.true.; model%tol=1.e6_dp
    end if
    select case(j)
    case(9,10); z=model%fit_transform(x,y,reg)
    case(11); z=model%fit_transform(x,y,elastic)
    case(12); z=model%fit_transform(x,y,sparse)
    case default; z=model%fit_transform(x,y,ols)
    end select
    write(prefix,'(a,i0)') 'case',j
    call emit('_support',real(merge(1,0,model%get_support()),dp))
    call emit('_shape',real([model%n_features_in_,model%n_features_to_select_],dp))
    call matrix('_train',z); call matrix('_query',model%transform(x+.137_dp))
    call matrix('_inverse',model%inverse_transform(z))
  end do
  close(output)
contains
  subroutine emit(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') trim(prefix)//suffix,values
  end subroutine
  subroutine matrix(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:,:)
    call emit(suffix,reshape(values,[size(values)]))
  end subroutine
end program
