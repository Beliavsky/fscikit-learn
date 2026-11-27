program reference_incremental_sparse_driver
  use sklearn_common,only:dp
  use sklearn_incremental_pca,only:incremental_pca
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(incremental_pca)::model
  type(csr_matrix)::s,t
  real(dp),allocatable::x(:,:),q(:,:),z(:,:)
  integer::n,p,m,k,b,w,train,query,i,j,e,input,output
  character(512)::path
  call get_command_argument(1,path);open(newunit=input,file=trim(path),status='old')
  read(input,*)n,p,m,k,b,w,train,query
  allocate(x(n,p),q(m,p))
  do i=1,n
    read(input,*)x(i,:)
  end do
  do i=1,m
    read(input,*)q(i,:)
  end do
  close(input);model%n_components=k;model%batch_size=b;model%whiten=w==1
  ! Fit once first, so every case also exercises reset of learned state.
  call model%fit(x)
  if(train==0)then
    z=model%fit_transform(x)
  else
    s=storage(x,train);z=model%fit_transform(s)
  end if
  call get_command_argument(2,path);open(newunit=output,file=trim(path),status='replace')
  call matrix('components',model%components_);call vector('mean',model%mean_);call vector('var',model%var_)
  call vector('singular',model%singular_values_);call vector('variance',model%explained_variance_)
  call vector('ratio',model%explained_variance_ratio_);call vector('noise',[model%noise_variance_])
  call matrix('train',z);call matrix('inverse',model%inverse_transform(z))
  if(query==0)then
    z=model%transform(q)
  else
    t=storage(q,query);z=model%transform(t)
  end if
  call matrix('query',z)
  call vector('shape',real([model%n_samples_seen_,model%n_features_in_,model%n_components_,model%batch_size_],dp))
  close(output)
contains
  function storage(a,kind)result(c)
    real(dp),intent(in)::a(:,:)
    integer,intent(in)::kind
    type(csr_matrix)::c
    integer::i,j,e,n,p
    if(kind==1)then
      c=csr_from_dense(a,store_zeros=.true.);return
    end if
    n=size(a,1);p=size(a,2);c%n_rows=n;c%n_cols=p
    allocate(c%indptr(n+1),c%indices(3*n*p),c%data(3*n*p));e=1;c%indptr(1)=1
    do i=1,n
      do j=p,1,-1
        c%indices(e:e+2)=j;c%data(e:e+2)=[.6_dp*a(i,j),.4_dp*a(i,j),0._dp];e=e+3
      end do
      c%indptr(i+1)=e
    end do
  end function
  subroutine vector(name,a)
    character(*),intent(in)::name
    real(dp),intent(in)::a(:)
    write(output,'(a,*(1x,es25.16e3))')name,a
  end subroutine
  subroutine matrix(name,a)
    character(*),intent(in)::name
    real(dp),intent(in)::a(:,:)
    call vector(name,reshape(a,[size(a)]))
  end subroutine
end program
