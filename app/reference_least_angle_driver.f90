program reference_least_angle_driver
  use sklearn, only: dp,lars,lasso_lars,lars_path,lars_path_result
  implicit none
  type(lars) :: model
  type(lasso_lars) :: sparse
  type(lars_path_result) :: path_result
  real(dp), allocatable :: x(:,:),y(:)
  integer :: n,p,i,j,input,output,limit
  real(dp) :: minimum
  character(8) :: method
  logical :: positive
  character(512) :: path
  character(24) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n))
  do i=1,n
    read(input,*) x(i,:),y(i)
  end do
  close(input); call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,20
    write(prefix,'(a,i0)') 'case',j
    if(j<=5) then
      model=lars()
      select case(j)
      case(2); model%n_nonzero_coefs=1
      case(3); model%n_nonzero_coefs=3; model%fit_intercept=.false.
      case(4); model%fit_path=.false.
      case(5); model%n_nonzero_coefs=1000
      end select
      call model%fit(x,y); call emit_model(model)
    else if(j<=14) then
      sparse=lasso_lars()
      select case(j)
      case(6); sparse%alpha=.1_dp
      case(7); sparse%alpha=.01_dp
      case(8); sparse%alpha=0
      case(9); sparse%alpha=.03_dp; sparse%positive=.true.
      case(10); sparse%alpha=0; sparse%positive=.true.
      case(11); sparse%alpha=1000
      case(12); sparse%alpha=0; sparse%max_iter=2
      case(13); sparse%alpha=.02_dp; sparse%fit_path=.false.
      case(14); sparse%alpha=.05_dp; sparse%fit_intercept=.false.
      end select
      call sparse%fit(x,y); call emit_model(sparse)
    else
      method='lasso'; if(j==15.or.j==19) method='lar'
      minimum=0; if(j==17) minimum=.03_dp
      if(j==20) minimum=.001_dp
      limit=500; if(j==19) limit=0
      if(j==20) limit=1
      positive=j==18
      path_result=lars_path(x,y,method=method,alpha_min=minimum,max_iter=limit,positive=positive)
      call vector('_alphas',path_result%alphas); call vector('_active',real(path_result%active,dp))
      call matrix('_path',path_result%coefficients); call vector('_iterations',real([path_result%n_iter],dp))
    end if
  end do
  close(output)
contains
  subroutine emit_model(estimator)
    class(lars), intent(in) :: estimator
    call vector('_coef',estimator%coef_); call vector('_intercept',[estimator%intercept_])
    call vector('_shape',real([estimator%n_features_in_,estimator%n_iter_],dp))
    call vector('_predict',estimator%predict(x)); call vector('_query',estimator%predict(x+.137_dp))
    call vector('_score',[estimator%score(x,y)]); call vector('_alphas',estimator%alphas_)
    if(estimator%fit_path) then
      call vector('_active',real(estimator%active_,dp)); call matrix('_path',estimator%coef_path_)
    end if
  end subroutine
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
