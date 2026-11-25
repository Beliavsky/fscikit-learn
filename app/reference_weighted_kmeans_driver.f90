program reference_weighted_kmeans_driver
  use sklearn_common,only:dp
  use sklearn_cluster,only:kmeans,k_means,kmeans_result
  use sklearn_sparse,only:csr_matrix
  implicit none
  type(kmeans)::model
  type(kmeans_result)::result
  type(csr_matrix)::sx,sq
  real(dp),allocatable::x(:,:),query(:,:),w(:),qw(:),initial(:,:),distance(:,:)
  integer,allocatable::labels(:)
  integer::n,p,k,m,train_sparse,query_sparse,has_weights,has_init,n_init,limit,seed,i,status,nnz,route
  real(dp)::tol,score
  character(16)::method
  do
    read(*,*,iostat=status)n,p,k,m,train_sparse,query_sparse,has_weights,has_init,n_init,limit,seed,tol,method,route
    if(status/=0)exit
    allocate(x(n,p),query(m,p),w(n),qw(m))
    read(*,*)w
    read(*,*)qw
    if(has_init==1)then
      allocate(initial(k,p))
      do i=1,k
        read(*,*)initial(i,:)
      end do
    end if
    if(train_sparse==0)then
      do i=1,n
        read(*,*)x(i,:)
      end do
    else
      call read_csr(sx,n,p)
    end if
    if(query_sparse==0)then
      do i=1,m
        read(*,*)query(i,:)
      end do
    else
      call read_csr(sq,m,p)
    end if
    model%n_clusters=k;model%n_init=n_init;model%max_iter=limit;model%random_state=seed;model%tol=tol;model%init=method
    if(allocated(model%initial_centers))deallocate(model%initial_centers)
    if(has_init==1)model%initial_centers=initial
    if(train_sparse==0)then
      if(route==0)then
        call model%fit(x,sample_weight=w)
      else
        result=k_means(x,k,sample_weight=w,init_centers=initial,init=trim(method),n_init=n_init, &
          max_iter=limit,tol=tol,random_state=seed)
        call assign_result()
      end if
    else
      if(route==0)then
        call model%fit(sx,sample_weight=w)
      else
        result=k_means(sx,k,sample_weight=w,init_centers=initial,init=trim(method),n_init=n_init, &
          max_iter=limit,tol=tol,random_state=seed)
        call assign_result()
      end if
    end if
    if(query_sparse==0)then
      labels=model%predict(query);distance=model%transform(query);score=model%score(query,qw)
    else
      labels=model%predict(sq);distance=model%transform(sq);score=model%score(sq,qw)
    end if
    write(*,*)model%n_iter_,model%n_features_in_,model%n_init_effective_
    write(*,'(*(es25.17e3,1x))')reshape(model%cluster_centers_,[k*p])
    write(*,*)model%labels_
    write(*,'(es25.17e3)')model%inertia_
    write(*,*)labels
    write(*,'(*(es25.17e3,1x))')reshape(distance,[m*k])
    write(*,'(es25.17e3)')score
    deallocate(x,query,w,qw,labels,distance)
    if(allocated(initial))deallocate(initial)
    if(train_sparse==1)deallocate(sx%indptr,sx%indices,sx%data)
    if(query_sparse==1)deallocate(sq%indptr,sq%indices,sq%data)
  end do
contains
  subroutine read_csr(a,n,p)
    type(csr_matrix),intent(out)::a
    integer,intent(in)::n,p
    integer::nnz
    a%n_rows=n;a%n_cols=p;read(*,*)nnz
    allocate(a%indptr(n+1),a%indices(nnz),a%data(nnz))
    read(*,*)a%indptr
    read(*,*)a%indices
    read(*,*)a%data
  end subroutine
  subroutine assign_result()
    model%cluster_centers_=result%centers;model%labels_=result%labels;model%inertia_=result%inertia
    model%n_iter_=result%n_iter;model%n_features_in_=p;model%n_init_effective_=result%n_init
  end subroutine
end program
