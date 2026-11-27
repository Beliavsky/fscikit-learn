program reference_glm_newton_paths_driver
  use sklearn_glm_newton,only:glm_newton_solve
  use sklearn_common,only:dp
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  integer::input,output,cases,c,n,p,intercept,repr,cap,i,j,k,iterations
  real(dp)::power,alpha,tol
  character(16)::link
  character(60)::message
  character(1024)::path
  logical::converged,before(size(ieee_all)),after(size(ieee_all)),hm(size(ieee_all)),ha(size(ieee_all))
  call get_command_argument(1,path);open(newunit=input,file=trim(path),status='old')
  call get_command_argument(2,path);open(newunit=output,file=trim(path),status='replace')
  read(input,*)cases
  do c=1,cases
    block
      real(dp),allocatable::x(:,:),y(:),w(:),point(:)
      type(csr_matrix)::csr
      read(input,*)n,p,power,link,alpha,tol,cap,intercept,repr
      allocate(x(n,p),y(n),w(n),point(p+intercept));read(input,*)point
      do i=1,n
        read(input,*)x(i,:),y(i),w(i)
      end do
      if(repr==1)csr=csr_from_dense(x)
      if(repr==2)then
        csr%n_rows=n;csr%n_cols=p
        allocate(csr%indptr(n+1),csr%indices(2*n*p+n),csr%data(2*n*p+n));k=1;csr%indptr(1)=1
        do i=1,n
          do j=p,1,-1
            csr%indices(k:k+1)=j;csr%data(k:k+1)=[.25_dp*x(i,j),.75_dp*x(i,j)];k=k+2
          end do
          csr%indices(k)=1;csr%data(k)=0;k=k+1;csr%indptr(i+1)=k
        end do
      end if
      do i=1,size(ieee_all)
        call ieee_get_flag(ieee_all(i),before(i));call ieee_get_halting_mode(ieee_all(i),hm(i))
      end do
      if(repr==0)then
        call glm_newton_solve(point,y,w,power,link,alpha,tol,cap,iterations,converged,message,dense=x)
      else
        call glm_newton_solve(point,y,w,power,link,alpha,tol,cap,iterations,converged,message,sparse=csr)
      end if
      do i=1,size(ieee_all)
        call ieee_get_flag(ieee_all(i),after(i));call ieee_get_halting_mode(ieee_all(i),ha(i))
      end do
      write(output,'(*(es25.16e3,1x))')point
      write(output,*)iterations,merge(1,0,converged),merge(1,0,all(before.eqv.after).and.all(hm.eqv.ha))
      write(output,'(a)')trim(message)
    end block
  end do
  close(input);close(output)
end program
