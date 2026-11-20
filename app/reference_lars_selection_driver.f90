program reference_lars_selection_driver
  use sklearn, only: dp,lars,lars_cv,lasso_lars_cv,lasso_lars_ic
  implicit none
  type(lars_cv) :: lar
  type(lasso_lars_cv) :: sparse
  type(lasso_lars_ic) :: information
  real(dp), allocatable :: x(:,:),y(:)
  integer :: input,output,n,p,i,j
  character(512) :: path
  character(24) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n))
  do i=1,n
    read(input,*) x(i,:),y(i)
  end do
  close(input); call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,18
    write(prefix,'(a,i0)') 'case',j
    if(j<=4) then
      lar=lars_cv(); lar%cv=3
      select case(j)
      case(2); lar%cv=4; lar%fit_intercept=.false.; lar%max_n_alphas=4
      case(3); lar%max_iter=2
      case(4); lar%max_iter=0
      end select
      if(n-(n+lar%cv-1)/lar%cv<p+merge(1,0,lar%fit_intercept).and. &
        lar%max_iter>=n-(n+lar%cv-1)/lar%cv) lar%eps=.1_dp
      call lar%fit(x,y); call emit_model(lar); call emit_cv(lar)
    else if(j<=10) then
      sparse=lasso_lars_cv(); sparse%cv=3
      select case(j)
      case(6); sparse%cv=4; sparse%max_n_alphas=4
      case(7); sparse%positive=.true.
      case(8); sparse%positive=.true.; sparse%fit_intercept=.false.; sparse%cv=4; sparse%max_n_alphas=6
      case(9); sparse%max_iter=3
      case(10); sparse%max_n_alphas=1
      end select
      call sparse%fit(x,y); call emit_model(sparse); call emit_cv(sparse)
    else
      information=lasso_lars_ic()
      if(j==12.or.j==14.or.j==16.or.j==18) information%criterion='bic'
      if(j==14) information%fit_intercept=.false.
      if(j==15.or.j==16) information%positive=.true.
      select case(j)
      case(13,14); information%noise_variance=.7_dp
      case(16); information%noise_variance=.4_dp
      case(17); information%noise_variance=.7_dp; information%max_iter=2
      case(18); information%noise_variance=.7_dp; information%max_iter=0
      end select
      if(n<=p+merge(1,0,information%fit_intercept)) information%noise_variance=.7_dp
      call information%fit(x,y); call emit_model(information)
      call vector('_alpha',[information%alpha_]); call vector('_criterion',information%criterion_)
      call vector('_noise',[information%noise_variance_])
    end if
  end do
  close(output)
contains
  subroutine emit_model(model)
    class(lars), intent(in) :: model
    call vector('_coef',model%coef_); call vector('_intercept',[model%intercept_])
    call vector('_shape',real([model%n_features_in_,model%n_iter_],dp))
    call vector('_predict',model%predict(x)); call vector('_query',model%predict(x+.137_dp))
    call vector('_score',[model%score(x,y)]); call vector('_alphas',model%alphas_)
  end subroutine
  subroutine emit_cv(model)
    class(lars_cv), intent(in) :: model
    call vector('_alpha',[model%alpha_]); call vector('_cv_alphas',model%cv_alphas_)
    call matrix('_mse',model%mse_path_); call matrix('_path',model%coef_path_)
    call vector('_active',real(model%active_,dp))
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
