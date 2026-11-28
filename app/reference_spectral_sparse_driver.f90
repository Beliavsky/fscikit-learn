program reference_spectral_sparse_driver
  use sklearn_common,only:dp
  use sklearn_spectral,only:spectral_embedding
  use sklearn_cluster,only:spectral_clustering
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(spectral_embedding)::embedding
  type(spectral_clustering)::cluster
  type(csr_matrix)::s
  real(dp),allocatable::x(:,:),z(:,:)
  integer,allocatable::labels(:)
  integer::n,p,k,algorithm,affinity,kind,neighbors,i,input,output
  character(32)::name
  character(512)::path
  call get_command_argument(1,path);open(newunit=input,file=trim(path),status='old')
  read(input,*)n,p,k,algorithm,affinity,kind,neighbors
  allocate(x(n,p))
  do i=1,n
    read(input,*)x(i,:)
  end do
  close(input)
  select case(affinity)
  case(0);name='rbf'
  case(1);name='nearest_neighbors'
  case(2);name='precomputed'
  case(3);name='precomputed_nearest_neighbors'
  end select
  if(kind>0)s=storage(x,kind)
  call get_command_argument(2,path);open(newunit=output,file=trim(path),status='replace')
  if(algorithm==0)then
    embedding%n_components=k;embedding%affinity=name;embedding%gamma=.4_dp;embedding%n_neighbors=neighbors
    if(kind==0)then
      z=embedding%fit_transform(x)
    else
      z=embedding%fit_transform(s)
    end if
    call matrix('affinity',embedding%affinity_matrix_);call matrix('embedding',z)
    write(output,'(a,1x,a)')'effective_affinity',trim(embedding%affinity)
  else
    cluster%n_clusters=k;cluster%affinity=name;cluster%gamma=.4_dp;cluster%n_neighbors=neighbors
    cluster%random_state=21;cluster%n_init=3
    if(algorithm==1)cluster%assign_labels='cluster_qr'
    if(kind==0)then
      labels=cluster%fit_predict(x)
    else
      labels=cluster%fit_predict(s)
    end if
    call matrix('affinity',cluster%affinity_matrix_);call matrix('embedding',cluster%embedding_)
    call vector('labels',real(labels,dp))
    write(output,'(a,1x,a)')'effective_affinity',trim(cluster%affinity)
  end if
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
