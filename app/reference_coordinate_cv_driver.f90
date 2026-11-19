program reference_coordinate_cv_driver
  use sklearn, only: dp,elastic_net_cv,lasso_cv
  implicit none
  type(elastic_net_cv) :: model
  type(lasso_cv) :: sparse
  real(dp), allocatable :: x(:,:),y(:),train(:,:)
  integer :: input,output,n,p,i,j
  character(512) :: path
  character(24) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n))
  do i=1,n
    read(input,*) x(i,:),y(i)
  end do
  close(input); call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,9
    model=elastic_net_cv(); model%alphas=[.1_dp,1._dp,.01_dp,.3_dp]; model%tol=1.e-12_dp
    model%cv=3; model%n_alphas=12; train=x
    select case(j)
    case(2); model%l1_ratios=[.2_dp,.7_dp,1._dp]
    case(3); deallocate(model%alphas); model%l1_ratios=[.3_dp,.8_dp]; model%eps=.02_dp
    case(4); model%fit_intercept=.false.; model%cv=4
    case(5); deallocate(model%alphas); model%n_alphas=1
    case(6); train(:,p)=train(:,1); model%l1_ratio=.7_dp
    end select
    write(prefix,'(a,i0)') 'case',j
    if(j<=6) then
      call model%fit(train,y); call outputs(model,train)
    else
      sparse=lasso_cv(); sparse%alphas=model%alphas; sparse%tol=1.e-12_dp; sparse%cv=3; sparse%n_alphas=12
      if(j==8) deallocate(sparse%alphas)
      if(j==9) sparse%fit_intercept=.false.
      call sparse%fit(train,y); call outputs(sparse,train)
    end if
  end do
  close(output)
contains
  subroutine outputs(fit,train)
    class(elastic_net_cv), intent(in) :: fit
    real(dp), intent(in) :: train(:,:)
    call vector('_coef',fit%coef_); call vector('_intercept',[fit%intercept_])
    call vector('_selected',[fit%alpha_,fit%l1_ratio_,fit%best_mse_])
    call vector('_alphas',reshape(fit%alphas_,[size(fit%alphas_)]))
    call vector('_path',reshape(fit%mse_path_,[size(fit%mse_path_)]))
    call vector('_train',fit%predict(train)); call vector('_query',fit%predict(train+.137_dp))
    call vector('_score',[fit%score(train,y)])
  end subroutine
  subroutine vector(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') trim(prefix)//suffix,values
  end subroutine
end program
