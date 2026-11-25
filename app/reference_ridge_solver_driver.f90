program reference_ridge_solver_driver
  use sklearn_common,only:dp
  use sklearn_ridge_solver
  use sklearn_sparse,only:csr_from_dense
  implicit none
  real(dp),allocatable::x(:,:),y(:,:),alpha(:),weights(:)
  type(ridge_solver_options)::options
  type(ridge_solution)::fit
  integer::n,p,t,i,j,count,layout,weighted,positive,ios
  do
    read(*,*,iostat=ios)n,p,t,count,layout,weighted
    if(ios/=0)exit
    read(*,*)options%solver,options%tol,options%max_iter,positive
    options%positive=positive/=0
    allocate(x(n,p),y(n,t),alpha(count),weights(n))
    do i=1,n
      read(*,*)x(i,:)
    end do
    do i=1,n
      read(*,*)y(i,:)
    end do
    read(*,*)alpha
    read(*,*)weights
    if(layout==0)then
      if(weighted==0)then
        fit=ridge_regression(x,y,alpha,options)
      else
        fit=ridge_regression(x,y,alpha,options,weights)
      end if
    else
      if(weighted==0)then
        fit=ridge_regression(csr_from_dense(x),y,alpha,options)
      else
        fit=ridge_regression(csr_from_dense(x),y,alpha,options,weights)
      end if
    end if
    do j=1,t
      write(*,'(*(es26.17e3,1x))')fit%coefficients(j,:)
    end do
    write(*,'(*(i0,1x))')fit%n_iter
    deallocate(x,y,alpha,weights)
  end do
end program
