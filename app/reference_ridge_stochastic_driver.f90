program reference_ridge_stochastic_driver
  use sklearn_common,only:dp
  use sklearn_ridge_solver
  use sklearn_sparse,only:csr_from_dense,csr_from_arrays,csr_matrix
  use sklearn_common,only:require
  implicit none
  real(dp),allocatable::x(:,:),y(:,:),alpha(:),weights(:)
  type(ridge_solver_options)::options
  type(ridge_solution)::fit
  integer::n,p,t,i,j,count,layout,weighted,positive,ios,intercept,k,pos
  type(csr_matrix)::sparse_x
  integer,allocatable::ptr(:),indices(:)
  real(dp),allocatable::data(:),xcopy(:,:),ycopy(:,:),acopy(:),wcopy(:)
  do
    read(*,*,iostat=ios)n,p,t,count,layout,weighted
    if(ios/=0)exit
    read(*,*)options%solver,options%tol,options%max_iter,positive,options%random_state,intercept
    options%positive=positive/=0;options%return_intercept=intercept/=0
    allocate(x(n,p),y(n,t),alpha(count),weights(n))
    do i=1,n
      read(*,*)x(i,:)
    end do
    do i=1,n
      read(*,*)y(i,:)
    end do
    read(*,*)alpha
    read(*,*)weights
    xcopy=x;ycopy=y;acopy=alpha;wcopy=weights
    if(layout/=0)then
      if(layout==1)then
        sparse_x=csr_from_dense(x)
      else if(layout==4)then
        sparse_x=csr_from_dense(x,store_zeros=.true.)
      else
        allocate(ptr(n+1),indices(2*n*p),data(2*n*p));pos=1;ptr(1)=1
        do i=1,n
          do j=p,1,-1
            if(x(i,j)==0)cycle
            indices(pos)=j
            if(layout==3)then
              data(pos)=.4_dp*x(i,j);pos=pos+1
              indices(pos)=j;data(pos)=.6_dp*x(i,j)
            else
              data(pos)=x(i,j)
            end if
            pos=pos+1
          end do
          ptr(i+1)=pos
        end do
        sparse_x=csr_from_arrays(n,p,ptr,indices(:pos-1),data(:pos-1))
        deallocate(ptr,indices,data)
      end if
    end if
    if(layout==0)then
      if(weighted==0)then
        fit=ridge_regression(x,y,alpha,options)
      else
        fit=ridge_regression(x,y,alpha,options,weights)
      end if
    else
      if(weighted==0)then
        fit=ridge_regression(sparse_x,y,alpha,options)
      else
        fit=ridge_regression(sparse_x,y,alpha,options,weights)
      end if
    end if
    call require(all(xcopy==x).and.all(ycopy==y).and.all(acopy==alpha).and.all(wcopy==weights),'ridge modified input')
    do j=1,t
      write(*,'(*(es26.17e3,1x))')fit%coefficients(j,:)
    end do
    write(*,'(*(i0,1x))')fit%n_iter
    write(*,'(*(es26.17e3,1x))')fit%intercept
    deallocate(x,y,alpha,weights)
  end do
end program
