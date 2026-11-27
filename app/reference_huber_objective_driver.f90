program reference_huber_objective_driver
  use sklearn_common,only:dp
  use sklearn_huber_solver,only:huber_loss_gradient
  use sklearn_sparse,only:csr_matrix,csr_from_arrays
  implicit none
  integer::cases,c,n,p,q,stored
  real(dp)::epsilon,alpha,loss
  real(dp),allocatable::x(:,:),y(:),weight(:),point(:),g(:),data(:)
  integer,allocatable::ptr(:),indices(:)
  type(csr_matrix)::csr
  read(*,*)cases
  do c=1,cases
    read(*,*)n,p,q,stored,epsilon,alpha
    allocate(y(n),weight(n),point(q),g(q))
    if(stored<0)then
      allocate(x(n,p));read(*,*)x
    else
      allocate(ptr(n+1),indices(stored),data(stored));read(*,*)ptr;read(*,*)indices;read(*,*)data
      csr=csr_from_arrays(n,p,ptr,indices,data)
    end if
    read(*,*)y;read(*,*)weight;read(*,*)point
    if(stored<0)then
      call huber_loss_gradient(point,y,weight,epsilon,alpha,loss,g,dense=x,dense_fortran=stored==-2)
      deallocate(x)
    else
      call huber_loss_gradient(point,y,weight,epsilon,alpha,loss,g,sparse=csr)
      deallocate(ptr,indices,data)
    end if
    print '(*(es25.16e3,1x))',loss
    print '(*(es25.16e3,1x))',g
    deallocate(y,weight,point,g)
  end do
end program
