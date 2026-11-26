program reference_nearest_neighbors_driver
 use sklearn_common,only:dp
 use sklearn_neighbors,only:nearest_neighbors
 use sklearn_sparse,only:csr_matrix
 implicit none
 type(nearest_neighbors)::model
 type(csr_matrix)::graph
 real(dp),allocatable::x(:,:),q(:,:),d(:,:)
 integer,allocatable::ids(:,:)
 real(dp)::power,radius
 integer::n,p,m,k,fit_k,leaf,family,implicit_query,exclude,i,status
 character(16)::metric,algorithm
 do
  read(*,*,iostat=status)n,p,m,k,fit_k,leaf,family,implicit_query,exclude,metric,power,algorithm,radius
  if(status/=0)exit
  allocate(x(n,p),q(m,p))
  do i=1,n;read(*,*)x(i,:);end do
  do i=1,m;read(*,*)q(i,:);end do
  model%metric=metric;model%p=power;model%algorithm=algorithm
  model%leaf_size=leaf;model%n_neighbors=fit_k;call model%fit(x)
  if(family==0)then
   if(implicit_query==1)then
    if(exclude==1)then
     call model%kneighbors(distances=d,indices=ids,n_neighbors=k)
    else
     call model%kneighbors(model%x_,d,ids,k)
    end if
   else
    call model%kneighbors(q,d,ids,k)
   end if
   write(*,'(a)')trim(model%fit_method_)
   write(*,'(*(es26.17e3,1x))')transpose(d)
   write(*,'(*(i0,1x))')transpose(ids)
  else
   if(implicit_query==1)then
    if(exclude==1)then
     graph=model%radius_neighbors_graph(radius=radius,mode='distance')
    else
     graph=model%radius_neighbors_graph(model%x_,radius,mode='distance')
    end if
   else
    graph=model%radius_neighbors_graph(q,radius,mode='distance')
   end if
   write(*,'(a)')trim(model%fit_method_)
   write(*,'(*(i0,1x))')graph%n_rows,graph%n_cols,graph%nnz()
   write(*,'(*(i0,1x))')graph%indptr
   write(*,'(*(i0,1x))')graph%indices
   write(*,'(*(es26.17e3,1x))')graph%data
  end if
  deallocate(x,q)
 end do
end program
