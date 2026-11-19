program reference_multitask_driver
  use sklearn, only: dp,multi_task_elastic_net,multi_task_lasso
  implicit none
  type(multi_task_elastic_net) :: net
  type(multi_task_lasso) :: lasso
  real(dp), allocatable :: x(:,:),y(:,:),target(:,:)
  integer :: n,p,t,i,j,input,output
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,t; allocate(x(n,p),y(n,t))
  do i=1,n
    read(input,*) x(i,:),y(i,:)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,8
    net=multi_task_elastic_net(); net%alpha=.08_dp; net%tol=1.e-10_dp; net%max_iter=20000; target=y
    select case(j)
    case(1)
      lasso=multi_task_lasso(); lasso%alpha=.08_dp; lasso%tol=1.e-10_dp; lasso%max_iter=20000
      call lasso%fit(x,y)
    case(2); net%l1_ratio=.2_dp
    case(3); net%l1_ratio=1
    case(4); net%fit_intercept=.false.
    case(5); net%alpha=100
    case(6); net%max_iter=1
    case(7)
      net%warm_start=.true.; net%alpha=.5_dp; call net%fit(x,y); net%alpha=.08_dp
    case(8); target=y(:,:1)
    end select
    write(prefix,'(a,i0)') 'case',j
    if(j==1) then
      call emit(lasso,target)
    else
      call net%fit(x,target); call emit(net,target)
    end if
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
  subroutine emit(model,targets)
    class(multi_task_elastic_net), intent(in) :: model
    real(dp), intent(in) :: targets(:,:)
    call matrix('_coef',model%coef_); call vector('_intercept',model%intercept_)
    call matrix('_train',model%predict(x)); call matrix('_query',model%predict(x+.137_dp))
    call vector('_score',[model%score(x,targets)]); call vector('_gap',[model%dual_gap_])
    call vector('_eps',[model%eps_]); call vector('_iterations',[real(model%n_iter_,dp)])
  end subroutine
end program
