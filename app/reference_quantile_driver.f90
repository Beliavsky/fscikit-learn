program reference_quantile_driver
  use sklearn_quantile, only: quantile_regressor
  use sklearn_common, only: dp
  implicit none
  type(quantile_regressor) :: model
  real(dp), allocatable :: x(:,:),y(:),w(:)
  integer :: n,p,i,input,output,intercept
  character(512) :: path
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,intercept
  read(input,*) model%quantile,model%alpha,model%tol,model%max_iter
  allocate(x(n,p),y(n),w(n))
  do i=1,n
    read(input,*) x(i,:),y(i),w(i)
  end do
  close(input); model%fit_intercept=intercept==1
  call model%fit(x,y,w)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  call vector('coef',model%coef_); call vector('intercept',[model%intercept_])
  call vector('predict',model%predict(x)); call vector('query',model%predict(x+.137_dp))
  call vector('score',[model%score(x,y)]); call vector('dual',model%dual_coef_)
  call vector('diagnostics',[model%objective_,model%dual_gap_,model%optimality_])
  call vector('state',real([model%n_iter_,model%n_features_in_,merge(1,0,model%converged_)],dp))
  close(output)
contains
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(output,'(a,1x,*(es25.16e3,1x))') name,a
  end subroutine
end program
