program reference_dummy_likelihood_driver
  use sklearn_common,only:dp
  use sklearn_stateless_preprocessing,only:add_dummy_feature
  use sklearn_covariance_functions,only:log_likelihood
  use sklearn_sparse,only:csr_matrix
  implicit none
  type(csr_matrix)::x,y
  real(dp),allocatable::a(:,:),b(:,:)
  real(dp)::value
  integer::mode,n,p,nnz,i,status
  do
    read(*,*,iostat=status)mode,n,p,value
    if(status/=0)exit
    if(mode==0)then
      allocate(a(n,p))
      do i=1,n
        read(*,*)a(i,:)
      end do
      b=add_dummy_feature(a,value)
      write(*,*)shape(b)
      write(*,'(*(es25.17e3,1x))')reshape(b,[size(b)])
      deallocate(a,b)
    else if(mode==1)then
      x%n_rows=n;x%n_cols=p;read(*,*)nnz
      allocate(x%indptr(n+1),x%indices(nnz),x%data(nnz))
      read(*,*)x%indptr
      read(*,*)x%indices
      read(*,*)x%data
      y=add_dummy_feature(x,value)
      write(*,*)y%n_rows,y%n_cols,y%nnz()
      write(*,*)y%indptr
      write(*,*)y%indices
      write(*,'(*(es25.17e3,1x))')y%data
      deallocate(x%indptr,x%indices,x%data)
    else
      allocate(a(p,p),b(p,p))
      do i=1,p
        read(*,*)a(i,:)
      end do
      do i=1,p
        read(*,*)b(i,:)
      end do
      write(*,'(es25.17e3)')log_likelihood(a,b)
      deallocate(a,b)
    end if
  end do
end program
