program reference_spectral_controls_driver
  use sklearn_common,only:dp
  use sklearn_spectral,only:spectral_embedding
  use sklearn_cluster,only:spectral_clustering
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(spectral_clustering)::model
  type(spectral_embedding)::embedding
  type(csr_matrix)::s
  real(dp),allocatable::x(:,:),z(:,:)
  integer,allocatable::labels(:)
  integer::n,p,k,c,assignment,kernel,kind,neighbors,i,input,output,estimator,connected
  real(dp)::gamma,degree,coef0
  character(32)::name
  character(512)::path
  call get_command_argument(1,path);open(newunit=input,file=trim(path),status='old')
  read(input,*)n,p,k,c,assignment,kernel,kind,neighbors,estimator
  read(input,*)gamma,degree,coef0
  allocate(x(n,p))
  do i=1,n
    read(input,*)x(i,:)
  end do
  close(input)
  select case(kernel)
  case(0);name='rbf'
  case(1);name='nearest_neighbors'
  case(2);name='precomputed'
  case(3);name='precomputed_nearest_neighbors'
  case(4);name='linear'
  case(5);name='poly'
  case(6);name='polynomial'
  case(7);name='sigmoid'
  case(8);name='cosine'
  case(9);name='laplacian'
  case(10);name='chi2'
  case(11);name='additive_chi2'
  end select
  if(kind>0)s=storage(x,kind)
  if(estimator==0)then
    model%n_clusters=k;model%n_components=c;model%affinity=name;model%gamma=gamma
    model%degree=degree;model%coef0=coef0;model%n_neighbors=neighbors;model%random_state=21;model%n_init=3
    if(assignment==1)model%assign_labels='cluster_qr'
    if(assignment==2)model%assign_labels='discretize'
    if(kind==0)then
      labels=model%fit_predict(x)
    else
      labels=model%fit_predict(s)
    end if
    z=model%embedding_;connected=model%n_connected_components_
  else
    embedding%n_components=c;embedding%affinity=name;embedding%gamma=gamma;embedding%n_neighbors=neighbors
    if(kind==0)then
      z=embedding%fit_transform(x)
    else
      z=embedding%fit_transform(s)
    end if
    connected=embedding%n_connected_components_
  end if
  call get_command_argument(2,path);open(newunit=output,file=trim(path),status='replace')
  if(estimator==0)then
    call matrix('affinity',model%affinity_matrix_);call vector('labels',real(labels,dp))
  else
    call matrix('affinity',embedding%affinity_matrix_)
  end if
  call matrix('embedding',z);call vector('shape',real([size(z,1),size(z,2),connected],dp))
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
