program reference_kernel_sparse_driver
  use sklearn_kernel_pca,only:kernel_pca
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_common,only:dp
  implicit none
  type(kernel_pca)::model
  type(csr_matrix)::s,t
  real(dp),allocatable::x(:,:),q(:,:),z(:,:),projected(:,:)
  integer::n,p,m,kernel,solver,train_storage,query_storage,i,unit,out
  character(1024)::path
  character(16),parameter::names(6)=[character(16)::'linear','rbf','poly','cosine','precomputed','sigmoid']
  call get_command_argument(1,path);open(newunit=unit,file=trim(path),status='old')
  read(unit,*)n,p,m,model%n_components,kernel,solver,train_storage,query_storage
  allocate(x(n,p),q(m,p))
  do i=1,n
    read(unit,*)x(i,:)
  end do
  do i=1,m
    read(unit,*)q(i,:)
  end do
  close(unit);model%kernel=names(kernel+1);model%gamma=.4_dp;model%degree=2;model%random_state=42
  model%iterated_power=1;if(solver==1)model%eigen_solver='randomized'
  if(train_storage==0)then
    z=model%fit_transform(x)
  else
    s=csr_from_dense(x,store_zeros=.true.);if(train_storage==2)s=raw(x)
    z=model%fit_transform(s)
  end if
  if(query_storage==0)then
    projected=model%transform(q)
  else
    t=csr_from_dense(q,store_zeros=.true.);if(query_storage==2)t=raw(q)
    projected=model%transform(t)
  end if
  call get_command_argument(2,path);open(newunit=out,file=trim(path),status='replace')
  write(out,*)size(model%eigenvalues_)
  call vector(model%eigenvalues_);call vector(reshape(z,[size(z)]))
  call vector(reshape(projected,[size(projected)]))
  write(out,*)model%n_features_in_,model%n_samples_fit_,merge(1,0,model%sparse_fit_), &
    merge(1,0,allocated(model%x_fit_)),merge(1,0,allocated(model%x_fit_csr_))
  close(out)
contains
  function raw(a)result(matrix)
    real(dp),intent(in)::a(:,:)
    type(csr_matrix)::matrix
    integer::r,j,k,ptr(size(a,1)+1),cols(3*size(a))
    real(dp)::data(3*size(a))
    k=1;ptr(1)=1
    do r=1,size(a,1)
      do j=size(a,2),1,-1
        cols(k:k+2)=j;data(k:k+2)=[.6_dp*a(r,j),.4_dp*a(r,j),0._dp];k=k+3
      end do
      ptr(r+1)=k
    end do
    matrix=csr_from_arrays(size(a,1),size(a,2),ptr,cols,data)
  end function
  subroutine vector(a)
    real(dp),intent(in)::a(:)
    write(out,'(*(es26.17e3,1x))')a
  end subroutine
end program
