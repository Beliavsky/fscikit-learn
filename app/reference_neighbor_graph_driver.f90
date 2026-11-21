program reference_neighbor_graph_driver
  use sklearn_neighbor_graph
  use sklearn_sparse
  use sklearn_common, only: dp
  implicit none
  type(kneighbors_transformer) :: kn
  type(radius_neighbors_transformer) :: rn
  type(csr_matrix) :: train,query,g
  real(dp), allocatable :: x(:,:),z(:,:),data(:)
  integer, allocatable :: ptr(:),idx(:)
  integer :: kind,storage,operation,k,n,p,m,i,nn
  real(dp) :: radius,power
  character(16) :: metric,mode
  read(*,*) kind,storage,operation,k,radius,power,n,p,m
  read(*,*) metric,mode
  kn%n_neighbors=k; rn%radius=radius
  kn%p=power; rn%p=power; kn%metric=metric; rn%metric=metric; kn%mode=mode; rn%mode=mode
  if(storage==0) then
    allocate(x(n,p),z(m,p))
    do i=1,n
      read(*,*) x(i,:)
    end do
    do i=1,m
      read(*,*) z(i,:)
    end do
    if(kind==0) then
      call kn%fit(x)
      select case(operation)
      case(0); g=kn%transform(z)
      case(1); g=kn%fit_transform(x)
      case(2); g=kn%transform()
      end select
    else
      call rn%fit(x)
      select case(operation)
      case(0); g=rn%transform(z)
      case(1); g=rn%fit_transform(x)
      case(2); g=rn%transform()
      end select
    end if
  else
    read(*,*) nn
    allocate(ptr(n+1),idx(nn),data(nn)); read(*,*) ptr; read(*,*) idx; read(*,*) data
    train=csr_from_arrays(n,p,ptr,idx,data); deallocate(ptr,idx,data)
    read(*,*) nn
    allocate(ptr(m+1),idx(nn),data(nn)); read(*,*) ptr; read(*,*) idx; read(*,*) data
    query=csr_from_arrays(m,p,ptr,idx,data)
    if(kind==0) then
      call kn%fit(train)
      select case(operation)
      case(0); g=kn%transform(query)
      case(1); g=kn%fit_transform(train)
      case(2); g=kn%transform()
      end select
    else
      call rn%fit(train)
      select case(operation)
      case(0); g=rn%transform(query)
      case(1); g=rn%fit_transform(train)
      case(2); g=rn%transform()
      end select
    end if
  end if
  print *,g%n_rows,g%n_cols,g%nnz()
  print *,g%indptr
  print *,g%indices
  print '( *(es25.16e3,1x))',g%data
end program
