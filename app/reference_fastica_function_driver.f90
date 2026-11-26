program reference_fastica_function_driver
 use sklearn_common,only:dp
 use sklearn_fast_ica,only:fastica,fastica_result
 implicit none
 type(fastica_result)::fit
 real(dp),allocatable::x(:,:),w(:,:)
 real(dp)::tol,alpha
 integer::n,p,k,seed,limit,initial,compute,status,id,i
 character(24)::algorithm,whiten,fun,solver
 id=0
 do
  read(*,*,iostat=status)n,p,k,seed,limit,initial,compute,tol,alpha,algorithm,whiten,fun,solver
  if(status<0)exit
  if(status/=0)error stop 'invalid ICA request'
  allocate(x(n,p))
  do i=1,n
   read(*,*)x(i,:)
  end do
  if(initial>0)then
   allocate(w(initial,initial))
   do i=1,initial
    read(*,*)w(i,:)
   end do
  end if
  if(fun=='custom_cube')then
   fit=fastica(x,n_components=k,random_state=seed,max_iter=limit,w_init=w,compute_sources=compute/=0, &
    tol=tol,alpha=alpha,algorithm=trim(algorithm),whiten=trim(whiten),whiten_solver=trim(solver),contrast_proc=cube_contrast)
  else
   fit=fastica(x,n_components=k,random_state=seed,max_iter=limit,w_init=w,compute_sources=compute/=0, &
    tol=tol,alpha=alpha,algorithm=trim(algorithm),whiten=trim(whiten),fun=trim(fun),whiten_solver=trim(solver))
  end if
  id=id+1;write(*,'(a,i0)')'CASE ',id
  write(*,*)fit%n_iter,merge(1,0,allocated(fit%whitening)),merge(1,0,allocated(fit%mean)), &
   merge(1,0,allocated(fit%sources))
  call matrix(fit%whitening);call matrix(fit%unmixing);call matrix(fit%sources)
  if(allocated(fit%mean))then
   write(*,'(*(es26.17e3,1x))')fit%mean
  else
   write(*,'(a)')''
  end if
  deallocate(x);if(allocated(w))deallocate(w)
 end do
contains
 subroutine cube_contrast(z,g,gprime)
  real(dp),intent(in)::z(:,:)
  real(dp),intent(out)::g(:,:),gprime(:)
  integer::row
  g=z**3
  do row=1,size(z,1)
   gprime(row)=sum(3*z(row,:)**2)/size(z,2)
  end do
 end subroutine
 subroutine matrix(a)
  real(dp),allocatable,intent(in)::a(:,:)
  if(allocated(a))then
   write(*,'(*(es26.17e3,1x))')a
  else
   write(*,'(a)')''
  end if
 end subroutine
end program
