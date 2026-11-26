program reference_graphical_function_driver
 use sklearn_covariance,only:graphical_lasso,graphical_lasso_result
 use sklearn_common,only:dp
 implicit none
 type(graphical_lasso_result)::fit
 real(dp),allocatable::cov(:,:)
 real(dp)::alpha,tol,enet_tol,eps
 integer::n,limit,i,status,id
 character(8)::mode
 id=0
 do
  read(*,*,iostat=status)n,alpha,limit,tol,enet_tol,eps,mode
  if(status<0)exit
  if(status/=0)error stop 'invalid graphical_lasso request'
  allocate(cov(n,n))
  do i=1,n
   read(*,*)cov(i,:)
  end do
  fit=graphical_lasso(cov,alpha,mode=mode,tol=tol,enet_tol=enet_tol,max_iter=limit,eps=eps)
  id=id+1;write(*,'(a,i0)')'CASE ',id
  write(*,*)fit%n_iter,size(fit%costs,1)
  write(*,'(*(es26.17e3,1x))')fit%covariance
  write(*,'(*(es26.17e3,1x))')fit%precision
  write(*,'(*(es26.17e3,1x))')fit%costs
  deallocate(cov)
 end do
end program
